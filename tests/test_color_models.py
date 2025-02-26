import pathlib
import random
import unittest
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from CDC.color_models import GaussianMixtureModelDistance, MahalanobisDistance, ReferencePixels

random.seed(1234)
np.random.seed(1234)

test_reference_pixel_image = np.astype(
    np.arange(0, 3 * 20 * 20, 1).reshape((3, 20, 20)) / (3 * 20 * 20) * 255, np.uint8
)
test_mask = np.astype(np.arange(0, 20 * 20, 1).reshape((1, 20, 20)) / (20 * 20) * 255, np.uint8)
test_bw_mask = np.where(test_mask > 100, 0, 255)
test_red_mask = test_reference_pixel_image
test_red_mask[0, :, :] = np.where(test_mask % 2 == 0, test_red_mask[0, :, :], 255)
test_red_mask[1, :, :] = np.where(test_mask % 2 == 0, test_red_mask[0, :, :], 0)
test_red_mask[2, :, :] = np.where(test_mask % 2 == 0, test_red_mask[0, :, :], 0)
test_wrong_size_mask = np.array([test_bw_mask, test_bw_mask])
test_too_small_mask = np.where(test_mask > 2, 0, 255)
test_image = np.array(
    [
        [[100, 50, 30], [30, 10, 70], [50, 45, 0]],
        [[50, 0, 0], [5, 20, 100], [60, 70, 60]],
        [[20, 30, 80], [50, 70, 10], [60, 80, 40]],
    ]
)
test_reference_pixels_values = np.array(
    [[5, 4, 6, 4, 5, 2, 3, 4, 5], [20, 20, 19, 21, 22, 19, 18, 20, 23], [100, 102, 101, 102, 99, 100, 102, 103, 98]]
)
test_mahal_res = np.array(
    [
        [
            [93.193112, 80.627193, 41.259647],
            [54.811516, 24.263431, 69.05152],
            [40.792615, 39.95395, 40.928231],
        ]
    ]
)
test_gmm_1_res = np.array(
    [
        [
            [69.891889, 60.466993, 30.938139],
            [41.103662, 18.186379, 51.784701],
            [30.587802, 29.958663, 30.689537],
        ]
    ]
)
test_gmm_2_res = np.array(
    [
        [
            [483.327411, 346.072851, 166.683859],
            [225.520187, 93.11609, 344.480443],
            [191.880031, 120.498269, 91.541404],
        ]
    ]
)


class TestReferencePixels(unittest.TestCase):
    def setUp(self) -> None:
        self.monkeypatch = pytest.MonkeyPatch()

    def test_reference_pixels(self) -> None:
        def get_mock_load_image(mask_to_use_as_test: NDArray[Any]) -> Callable[[Any, Any, Any, Any], Any]:
            def mock_load_image(self: Any, file_name: pathlib.Path, *args: Any, **kwargs: Any) -> NDArray[Any] | None:
                if file_name == pathlib.Path("reference"):
                    return test_reference_pixel_image
                elif file_name == pathlib.Path("annotated"):
                    return mask_to_use_as_test
                return None

            return mock_load_image

        with self.monkeypatch.context() as mp:
            # test red annotations
            mp.setattr(ReferencePixels, "load_image", get_mock_load_image(test_red_mask))
            ReferencePixels(reference=pathlib.Path("reference"), annotated=pathlib.Path("annotated"))
            # test black and white mask
            mp.setattr(ReferencePixels, "load_image", get_mock_load_image(test_bw_mask))
            ReferencePixels(reference=pathlib.Path("reference"), annotated=pathlib.Path("annotated"))
            # test mask of the wrong type
            mp.setattr(ReferencePixels, "load_image", get_mock_load_image(test_wrong_size_mask))
            with pytest.raises(TypeError):
                ReferencePixels(reference=pathlib.Path("reference"), annotated=pathlib.Path("annotated"))
            # test mask which selects to few pixels
            mp.setattr(ReferencePixels, "load_image", get_mock_load_image(test_too_small_mask))
            with pytest.raises(Exception, match=r"Not enough annotated pixels. Need at least \d+, but got \d+"):
                ReferencePixels(reference=pathlib.Path("reference"), annotated=pathlib.Path("annotated"))


class TestColorModels(unittest.TestCase):
    def setUp(self) -> None:
        self.monkeypatch = pytest.MonkeyPatch()

    def test_calculate_distance(self) -> None:
        def mock_reference_pixels_init(self: Any, *args: Any, **kwargs: Any) -> None:
            self.values = test_reference_pixels_values

        # test Mahalanobis distance calculations
        md = MahalanobisDistance(reference_pixels=test_reference_pixels_values, bands_to_use=(0, 1, 2))
        np.testing.assert_almost_equal(md.calculate_distance(test_image), test_mahal_res, decimal=6)
        # test Gaussian Mixture Model distance calculations with 1 cluster
        gmmd1 = GaussianMixtureModelDistance(
            reference_pixels=test_reference_pixels_values, bands_to_use=(0, 1, 2), n_components=1
        )
        np.testing.assert_almost_equal(gmmd1.calculate_distance(test_image), test_gmm_1_res, decimal=6)
        # test Gaussian Mixture Model distance calculations with 2 cluster
        gmmd2 = GaussianMixtureModelDistance(
            reference_pixels=test_reference_pixels_values, bands_to_use=(0, 1, 2), n_components=2
        )
        np.testing.assert_almost_equal(gmmd2.calculate_distance(test_image), test_gmm_2_res, decimal=6)

        # test bands_to_use and alpha_channel
        md2 = MahalanobisDistance.from_pixel_values(pixel_values=test_reference_pixels_values)
        assert md2.bands_to_use == (0, 1)
        md3 = MahalanobisDistance.from_pixel_values(pixel_values=test_reference_pixels_values, bands_to_use=(0, 1, 2))
        assert md3.bands_to_use == (0, 1, 2)
        md4 = MahalanobisDistance.from_pixel_values(pixel_values=test_reference_pixels_values, alpha_channel=1)
        assert md4.bands_to_use == (0, 2)
        with pytest.raises(ValueError, match=r"Bands have to be between 0 and \d+, but got -?\d+\."):
            MahalanobisDistance.from_pixel_values(pixel_values=test_reference_pixels_values, bands_to_use=[-1])
        with pytest.raises(ValueError, match=r"Bands have to be between 0 and \d+, but got -?\d+\."):
            MahalanobisDistance.from_pixel_values(pixel_values=test_reference_pixels_values, bands_to_use=[0, 2, 8])
        with pytest.raises(ValueError, match=r"Alpha channel have to be between -1 and \d+, but got -?\d+\."):
            MahalanobisDistance.from_pixel_values(pixel_values=test_reference_pixels_values, alpha_channel=-2)
        with pytest.raises(ValueError, match=r"Alpha channel have to be between -1 and \d+, but got -?\d+\."):
            MahalanobisDistance.from_pixel_values(pixel_values=test_reference_pixels_values, alpha_channel=8)
        # test from_pixel_values constructor
        md5 = MahalanobisDistance.from_pixel_values(pixel_values=test_reference_pixels_values, bands_to_use=(0, 1, 2))
        np.testing.assert_almost_equal(md5.calculate_distance(test_image), test_mahal_res, decimal=6)
        gmmd3 = GaussianMixtureModelDistance.from_pixel_values(
            pixel_values=test_reference_pixels_values,
            bands_to_use=(0, 1, 2),
            n_components=1,
        )
        np.testing.assert_almost_equal(gmmd3.calculate_distance(test_image), test_gmm_1_res, decimal=6)

        with self.monkeypatch.context() as mp:
            # test from_image_annotation constructor
            mp.setattr(ReferencePixels, "__init__", mock_reference_pixels_init)
            md6 = MahalanobisDistance.from_image_annotation(
                reference=pathlib.Path("reference"), annotated=pathlib.Path("annotated"), bands_to_use=(0, 1, 2)
            )
            np.testing.assert_almost_equal(md6.calculate_distance(test_image), test_mahal_res, decimal=6)
            gmmd4 = GaussianMixtureModelDistance.from_image_annotation(
                reference=pathlib.Path("reference"),
                annotated=pathlib.Path("annotated"),
                bands_to_use=(0, 1, 2),
                n_components=1,
            )
            np.testing.assert_almost_equal(gmmd4.calculate_distance(test_image), test_gmm_1_res, decimal=6)
