"""Interface protocols."""

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from torch import Tensor, nn


@runtime_checkable
class TorchNetwork(Protocol):
    """PyTorch Neural Network Model."""

    ...

    def children(self) -> Iterator[nn.Module]:
        """Return an iterator over model's immediate children modules."""
        ...

    def eval(self: "TorchNetwork") -> "TorchNetwork":
        """Set the model to evaluation mode."""
        ...

    def parameters(self, *, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Return an iterator over model's parameters."""
        ...

    def train(
        self: "TorchNetwork", mode: bool = True  # noqa: FBT001 FBT002
    ) -> "TorchNetwork":
        """Set the model to training mode."""
        ...


@runtime_checkable
class SklearnClassifier(Protocol):
    """Scikit-learn Classification Model."""

    ...

    def predict(self, features: Tensor) -> Tensor | tuple[Tensor, ...]:
        """Predict class from features."""
        ...

    def predict_proba(self, features: Tensor) -> Tensor | tuple[Tensor, ...]:
        """Predict class probabilities from features."""
        ...


Model = TorchNetwork | SklearnClassifier
