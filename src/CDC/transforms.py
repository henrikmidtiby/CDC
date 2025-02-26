"""Transform images when they are read."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np  # noqa: F401 # so numpy can be used in lambdas
from numpy.typing import NDArray


class BaseTransform(ABC):
    """Base class for transforms."""

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def transform(self, image: NDArray[Any]) -> NDArray[Any]:
        """
        Transform the image.

        Parameters
        ----------
        image
            The image to apply the transform to.
        """
        pass


class GammaTransform(BaseTransform):
    """
    Transform images with a gamma correction.

    Parameters
    ----------
    gamma
        The gamma value to use for the gamma correction. Must be positive.
    """

    def __init__(self, gamma: float) -> None:
        if not gamma > 0:
            raise ValueError("Gamma must be positive")
        self.gamma: float = gamma

    def transform(self, image: NDArray[Any]) -> NDArray[Any]:
        if np.any(image < 0):
            raise ValueError("Image can not have negative values in the Gamma Transform")
        gamma_corrected_image = image**self.gamma
        return gamma_corrected_image


class LambdaTransform(BaseTransform):
    """
    Transform images using an Lambda expression.

    Parameters
    ----------
    lambda_expression
        Either a function which takes the images and perform the transformation.
        Or a string in the form of a python lambda expression.

    Examples
    --------
    Using a function:

    >>> from CDC.transforms import LambdaTransform
    >>> def normalize(image):
    ...     return image / np.max(image)
    >>> transform = LambdaTransform(normalize)

    Using a string:

    >>> from CDC.transforms import LambdaTransform
    >>> lambda_str = "lambda im: im/np.min(im) + 50"
    >>> transform = LambdaTransform(lambda_str)
    """

    def __init__(self, lambda_expression: Callable[[NDArray[Any]], NDArray[Any]] | str) -> None:
        if isinstance(lambda_expression, str):
            if lambda_expression.startswith("lambda"):
                self.lambda_exp: Callable[[NDArray[Any]], NDArray[Any]] = eval(lambda_expression)
            else:
                raise ValueError("Lambda expression as string have to start with 'lambda'")
        else:
            self.lambda_exp = lambda_expression

    def transform(self, image: NDArray[Any]) -> NDArray[Any]:
        res_image = self.lambda_exp(image)
        if res_image.shape != image.shape:
            raise ValueError(
                f"Lambda expression may not change the image shape! input shape: {image.shape}, output shape: {res_image.shape}"
            )
        return self.lambda_exp(image)
