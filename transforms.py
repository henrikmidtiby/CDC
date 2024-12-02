from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np  # noqa: F401 # so numpy can be used in lambdas
from numpy.typing import NDArray


class BaseTransformer(ABC):
    """Base class for all color distance models."""

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def transform(self, image: NDArray[Any]) -> NDArray[Any]:
        pass


class GammaTransform(BaseTransformer):
    """Transform for gamma correction."""

    def __init__(self, gamma: float) -> None:
        self.gamma: float = gamma

    def transform(self, image: NDArray[Any]) -> NDArray[Any]:
        gamma_corrected_image = image**self.gamma
        return gamma_corrected_image


class LambdaTransform(BaseTransformer):
    """Transform images using an Lambda expression."""

    def __init__(self, lambda_expression: Callable[[NDArray[Any]], NDArray[Any]] | str) -> None:
        if isinstance(lambda_expression, str):
            if lambda_expression.startswith("lambda"):
                self.lambda_exp: Callable[[NDArray[Any]], NDArray[Any]] = eval(lambda_expression)
            else:
                raise Exception("Lambda expression as string have to start with 'lambda'")
        else:
            self.lambda_exp = lambda_expression

    def transform(self, image: NDArray[Any]) -> NDArray[Any]:
        res_image = self.lambda_exp(image)
        if res_image.shape != image.shape:
            raise Exception(
                f"Lambda expression may not change the image shape! input shape: {image.shape}, output shape: {res_image.shape}"
            )
        return self.lambda_exp(image)
