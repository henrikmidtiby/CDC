from abc import ABC, abstractmethod
from typing import Any

from numpy.typing import NDArray


class BaseTransformer(ABC):
    """Base class for all color distance models."""

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def transform(self, image: NDArray[Any]) -> NDArray[Any]:
        pass


class GammaCorrector(BaseTransformer):
    """Transform for gamma correction."""

    def __init__(self, gamma: float) -> None:
        self.gamma: float = gamma

    def transform(self, image: NDArray[Any]) -> NDArray[Any]:
        gamma_corrected_image = image**self.gamma
        return gamma_corrected_image
