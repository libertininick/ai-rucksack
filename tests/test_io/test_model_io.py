"""Tests for model IO functions."""

from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
from sklearn.naive_bayes import GaussianNB

from ai_rucksack.io import load_model, save_model
from ai_rucksack.utils.protocols import Model


def assert_models_equal(model1: Model, model2: Model) -> None:
    """Assert that two models are equal."""
    if isinstance(model1, torch.nn.Module) and isinstance(model2, torch.nn.Module):
        m1_dict = model1.state_dict()
        m2_dict = model2.state_dict()
    else:
        m1_dict = model1.__dict__
        m2_dict = model2.__dict__

    assert all(
        (
            np.allclose(v, m2_dict[k])
            if isinstance(v, np.ndarray | torch.Tensor)
            else v == m2_dict[k]
        )
        for k, v in m1_dict.items()
    )


@pytest.mark.parametrize("as_script", [False, True])
@pytest.mark.parametrize("parse_checksum", [False, True])
@pytest.mark.parametrize("append_checksum", [False, True])
@pytest.mark.parametrize("secret_key", [None, "key", b"\xa3\x9d"])
@pytest.mark.parametrize(
    "model", [torch.nn.Linear(1, 1), GaussianNB().fit([[1, 2], [3, 4]], [1, 0])]
)
def test_save_load_model(
    model: Model,
    secret_key: str | bytes | None,
    *,
    append_checksum: bool,
    parse_checksum: bool,
    as_script: bool,
) -> None:
    """Test saving and reloading a simple model."""
    with TemporaryDirectory() as temp_dir:
        # Save model
        model_receipt = save_model(
            model,
            temp_dir,
            secret_key=secret_key,
            append_checksum=append_checksum,
            as_script=as_script,
        )

        # Load model
        if append_checksum and not parse_checksum:
            # Check that a warning is raised that model was not validated
            with pytest.warns(
                RuntimeWarning,
                match=(
                    "No checksum provided; "
                    "deserializing model without validating integrity."
                ),
            ):
                loaded_model = load_model(
                    source=model_receipt.model_path,
                    checksum=None,
                    secret_key=secret_key,
                    parse_checksum=parse_checksum,
                )
        else:
            loaded_model = load_model(
                source=model_receipt.model_path,
                checksum=model_receipt.model_checksum if not append_checksum else None,
                secret_key=secret_key,
                parse_checksum=parse_checksum,
            )

        # Check that the loaded model is equal to the original model
        assert_models_equal(model, loaded_model)

        if as_script and isinstance(model, torch.nn.Module):
            # Check that the loaded model is a ScriptModule
            assert torch.jit.isinstance(loaded_model, torch.jit.ScriptModule)
