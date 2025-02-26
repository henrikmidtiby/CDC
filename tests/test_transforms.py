import unittest
from typing import Any

import numpy as np
import pytest
from numpy.random import default_rng
from numpy.typing import NDArray

from CDC.transforms import GammaTransform, LambdaTransform

test_float_image_0_1 = default_rng(1234).random((3, 5, 5))
test_float_image_neg1_1 = test_float_image_0_1 * 2 - 1
test_uint8_image = (test_float_image_0_1 * 255).astype(np.uint8)


class TestTransformers(unittest.TestCase):
    def test_gamma(self) -> None:
        # test Exceptions
        with pytest.raises(ValueError, match="Gamma must be positive"):
            GammaTransform(-1)
        with pytest.raises(ValueError, match="Image can not have negative values in the Gamma Transform"):
            GammaTransform(2.5).transform(test_float_image_neg1_1)
        # test 0 to 1 float images
        np.testing.assert_equal(GammaTransform(2).transform(test_float_image_0_1), test_float_image_0_1**2)
        np.testing.assert_equal(GammaTransform(0.3).transform(test_float_image_0_1), test_float_image_0_1**0.3)
        # test uint8 images
        np.testing.assert_equal(GammaTransform(4).transform(test_uint8_image), test_uint8_image**4)
        np.testing.assert_equal(GammaTransform(0.6).transform(test_uint8_image), test_uint8_image**0.6)

    def test_lambda(self) -> None:
        def lambda_test_func(image: NDArray[Any]) -> NDArray[Any]:
            new_image: NDArray[Any] = image / np.min(image)
            return new_image

        def lambda_test_wrong_params_func(image: NDArray[Any], power: float) -> NDArray[Any]:
            return image**power

        # test Exceptions
        with pytest.raises(ValueError, match="Lambda expression as string have to start with 'lambda'"):
            LambdaTransform("not a lambda expression: 2x^3")
        with pytest.raises(
            ValueError, match=r"Lambda expression may not change the image shape! input shape: (.*), output shape: (.*)"
        ):
            LambdaTransform("lambda x: np.min(x)").transform(test_float_image_0_1)
        with pytest.raises(TypeError):
            LambdaTransform("lambda x, y: x*y").transform(test_float_image_0_1)
        with pytest.raises(TypeError):
            LambdaTransform(lambda_test_wrong_params_func).transform(test_float_image_0_1)  # type: ignore[arg-type]
        # test 0 to 1 float images
        np.testing.assert_equal(
            LambdaTransform("lambda x: 3*x").transform(test_float_image_0_1), test_float_image_0_1 * 3
        )
        np.testing.assert_equal(
            LambdaTransform("lambda x: 2*x+5").transform(test_float_image_0_1), test_float_image_0_1 * 2 + 5
        )
        np.testing.assert_equal(
            LambdaTransform(lambda_test_func).transform(test_float_image_0_1),
            test_float_image_0_1 / np.min(test_float_image_0_1),
        )
        # test uint8 images
        np.testing.assert_equal(LambdaTransform("lambda x: 3*x").transform(test_uint8_image), test_uint8_image * 3)
        np.testing.assert_equal(
            LambdaTransform("lambda x: 2*x+5").transform(test_uint8_image), test_uint8_image * 2 + 5
        )
