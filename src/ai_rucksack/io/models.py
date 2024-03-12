"""I/O functions for saving and loading models to disk.

`torch.save` uses Python's `pickle` module to serialize objects. This is
insecure and can lead to arbitrary code execution. To mitigate this risk,
`load_model` and `save_model` functions use a checksum to validate the
integrity of the model file before deserializing it. The checksum can be
signed with a secret key to prevent tampering.

`load_model` and `save_model` functions can also be used to load and save
`sklearn` models.
"""

import hmac
import io
import re
from pathlib import Path
from warnings import warn

import msgspec
import torch

from ai_rucksack.utils.protocols import Model, TorchNetwork

CHECKSUM_PATTERN = re.compile(r"(?P<checksum>[A-Za-z0-9]{64})\.pts?$")
FILE_EXTENSIONS = (".pt", ".pts")


class ModelReceipt(msgspec.Struct, frozen=True):  # type: ignore
    """Receipt of a saved model.

    Attributes
    ----------
    model_path: str
        Path to saved model.
    model_checksum: str
        Checksum of saved model.
    signed: bool
        Flag of whether model checksum is signed with a secret key.
    digestmod: str
        Hash algorithm used for computing checksum.
    as_script: bool
        Flag of whether model was saved as TorchScript.
    """

    model_path: str
    model_checksum: str
    signed: bool
    digestmod: str
    as_script: bool

    def to_dict(self) -> dict[str, str | bool]:
        """Convert to serializable dictionary."""
        return msgspec.to_builtins(self)


def load_model(
    source: Path | str,
    *,
    checksum: str | None = None,
    secret_key: str | bytes | bytearray | None = None,
    digestmod: str = "sha256",
    parse_checksum: bool = True,
    map_location: str = "cpu",
) -> Model:
    """Load model from disk.

    Parameters
    ----------
    source: Path | str
        Model filepath.
    checksum: str | None, optional
        Checksum to validate model file.
        If `None` & `parse_checksum=True`, checksum is parsed from the model file path.
        (default = None)
    secret_key: str | bytes | bytearray | None, optional
        secret key for combining with model's hash before validating checksum
        and deserializing the model from bytes.
        (default = None)
    digestmod: str, optional
        Hash algorithm to use for computing checksum.
        (default = "sha256")
    parse_checksum: bool, optional
        Flag of whether to parse checksum from model file path.
        (default = True)
    map_location: str, optional
        Device to load model to (for TorchNetwork).
        NOTE: This parameter has no effect on non TorchNetwork models.
        (default = "cpu")

    Returns
    -------
    Model
        Instance of saved model

    Raises
    ------
    FileNotFoundError
        If model file not found.
    ValueError
        If checksums don't match.

    Notes
    -----
    - This function validates the integrity of the model file using a checksum
    before deserializing it.
    - If a checksum is not provided, the function will attempt to parse
    it from the model file path.
    - If `parse_checksum=False` and no checksum is provided, a warning about
    potential security risk is raised.
    - Extension of model file determines whether to load as TorchScript or not.
    """
    # Validate source
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Model file not found: {source}")

    if (model_checksum := checksum) is None and parse_checksum:
        # Parse checksum from model path
        m = CHECKSUM_PATTERN.search(str(source))
        if m is None:
            raise ValueError("No checksum parsed from model path.")
        model_checksum = m["checksum"]

    # Load model
    return _load_n_validate_model(
        source, model_checksum, secret_key, digestmod, map_location
    )


def save_model(
    model: Model,
    output_dir: Path | str,
    *,
    tag: str = "latest",
    secret_key: str | bytes | bytearray | None = None,
    digestmod: str = "sha256",
    append_checksum: bool = True,
    as_script: bool = False,
) -> ModelReceipt:
    """Save a model to disk in a specified directory.

    Parameters
    ----------
    model: Model
        Model to save.
    output_dir: Path | str
        Directory where to save model.
    tag: str, optional
        Tag to include in model filename.
        (default = "latest")
    secret_key: str | bytes | bytearray | None, optional
        Secret key for combining with model's hash before saving model to disk.
        (default = None)
    digestmod: str, optional
        Hash algorithm to use for computing checksum.
        (default = "sha256")
    append_checksum: bool, optional
        Flag of whether to append checksum to model filename.
        (default = True)
    as_script: bool, optional
        Flag of whether to save model (if TorchNetwork) in TorchScript format.
        (default = False)

    Returns
    -------
    ModelReceipt

    Example
    -------
    >>> import secrets
    >>> import tempfile
    >>> import torch
    >>> from ai_rucksack.io import save_model

    >>> model = torch.nn.Linear(1, 1)
    >>> key = secrets.token_bytes(16)
    >>> with tempfile.TemporaryDirectory() as model_dir:
    ...     receipt = save_model(model, model_dir, secret_key=key)

    Notes
    -----
    - By default, the concatenation of model's class name, the specified model tag and
    the model's sha256 checksum hex digest is used as the file name:
        - `<output_dir>/<classname>_<tag>_<checksum>.pt`
    - Models are saved with `.pts` extension for TorchScript format (`as_script=True`)
    - Otherwise models are saved with `.pt` extension (`as_script=False`)
    """
    # Save model to buffer
    buffer = io.BytesIO()
    if isinstance(model, TorchNetwork) and as_script:
        torch.jit.save(torch.jit.script(model), buffer)
        extension = "pts"
    else:
        torch.save(model, buffer)
        extension = "pt"

    # Get checksum from buffer
    model_checksum = hmac.new(
        _validate_key(secret_key), buffer.getvalue(), digestmod
    ).hexdigest()

    # Specify a path
    classname = model.__class__.__name__
    if append_checksum:
        file_name = f"{classname}_{tag}_{model_checksum}.{extension}"
    else:
        file_name = f"{classname}_{tag}.{extension}"
    model_path = Path(output_dir) / file_name

    # Save
    with open(model_path, "wb") as fp:
        fp.write(buffer.getvalue())

    # Return receipt
    return ModelReceipt(
        str(model_path), model_checksum, secret_key is not None, digestmod, as_script
    )


def _validate_key(key: str | bytes | bytearray | None) -> bytes | bytearray:
    """Validate and convert `key` to bytes or bytearray."""
    match key:
        case None:
            # No key, return empty bytes
            return bytearray()
        case str():
            # Encode string to bytes
            return key.encode("utf-8")
        case bytes() | bytearray():
            # Already bytes or bytearray
            return key
        case _:
            raise TypeError(f"Invalid `key` type: {type(key)}")


def _load_n_validate_model(
    source: Path,
    model_checksum: str | None,
    secret_key: str | bytes | bytearray | None,
    digestmod: str,
    map_location: str,
) -> Model:
    """Read model into a buffer, validate checksum, and load model instance."""
    with open(source, "rb") as fp:
        # Read model into buffer
        buffer = io.BytesIO(fp.read())

        # Validate checksum
        if model_checksum is None:
            # No checksum provided, issue warning
            warn(
                message=(
                    "No checksum provided; "
                    "deserializing model without validating integrity."
                ),
                category=RuntimeWarning,
                stacklevel=2,
            )
        else:
            # Compute model digest
            model_digest = hmac.new(
                _validate_key(secret_key), buffer.getvalue(), digestmod
            ).hexdigest()

            # Validate checksum and digest match
            if not hmac.compare_digest(model_checksum, model_digest):
                raise ValueError("Checksums don't match.")

        # Load model from buffer
        if source.suffix == ".pts":
            # Deserialize model as a `ScriptModule` using `torch.jit.load`
            model = torch.jit.load(buffer, map_location=map_location)
        else:
            # Deserialize model using `torch.load`
            if source.suffix not in FILE_EXTENSIONS:
                warn(
                    (
                        f"Unrecognized model file extension: {source.suffix}; "
                        "defaulting to `torch.load` for deserialization."
                    ),
                    stacklevel=2,
                )
            model = torch.load(buffer, map_location=map_location)

        # Validate is instance of Model
        if not isinstance(model, Model):  # type: ignore
            raise TypeError(f"Loaded model is not a Model instance: {type(model)}")

        return model
