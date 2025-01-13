import pathlib
import random
import unittest
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from OCDC.color_models import GaussianMixtureModelDistance, MahalanobisDistance, ReferencePixels

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
            [69.874696, 60.447119, 30.899279],
            [41.07442, 18.120193, 51.761494],
            [30.548496, 29.918531, 30.650361],
        ]
    ]
)
test_gmm_2_res = np.array(
    [
        [
            [483.326105, 346.071027, 166.680071],
            [225.517387, 93.109311, 344.478611],
            [191.876741, 120.49303, 91.534508],
        ]
    ]
)


class TestReferencePixels(unittest.TestCase):
    def setUp(self) -> None:
        self.monkeypatch = pytest.MonkeyPatch()

    def test_reference_pixels(self) -> None:
        def mock_load_reference_image(self: Any, *args: Any, **kwargs: dict[str, Any]) -> None:
            self.reference_image = test_reference_pixel_image

        def get_mock_load_mask(mask_to_use_as_mock: NDArray[Any]) -> Callable[[Any, Any, dict[str, Any]], None]:
            def mock_load_mask(self: Any, *args: Any, **kwargs: dict[str, Any]) -> None:
                self.mask = mask_to_use_as_mock

            return mock_load_mask

        with self.monkeypatch.context() as mp:
            mp.setattr(ReferencePixels, "_load_reference_image", mock_load_reference_image)
            # test mask with red annotations
            mp.setattr(ReferencePixels, "_load_mask", get_mock_load_mask(test_red_mask))
            ReferencePixels(
                reference=pathlib.Path("test"), annotated=pathlib.Path("test"), bands_to_use=(0, 1, 2), transform=None
            )
            # test bands_to_use is set correct and matches image
            rp_none_alpha_none = ReferencePixels(
                reference=pathlib.Path("test"),
                annotated=pathlib.Path("test"),
                bands_to_use=None,
                alpha_channel=None,
                transform=None,
            )
            assert rp_none_alpha_none.bands_to_use == (0, 1, 2)
            rp_01 = ReferencePixels(
                reference=pathlib.Path("test"), annotated=pathlib.Path("test"), bands_to_use=(0, 1), transform=None
            )
            assert rp_01.bands_to_use == (0, 1)
            rp_01_alpha_neg1 = ReferencePixels(
                reference=pathlib.Path("test"), annotated=pathlib.Path("test"), bands_to_use=None, transform=None
            )
            assert rp_01_alpha_neg1.bands_to_use == (0, 1)
            rp_02_alpha_1 = ReferencePixels(
                reference=pathlib.Path("test"),
                annotated=pathlib.Path("test"),
                bands_to_use=None,
                alpha_channel=1,
                transform=None,
            )
            assert rp_02_alpha_1.bands_to_use == (0, 2)
            # test alpha channel an bands_to_use raises exceptions if out of bounds
            with pytest.raises(ValueError, match=r"Bands have to be between 0 and \d+, but got -?\d+\."):
                ReferencePixels(
                    reference=pathlib.Path("test"), annotated=pathlib.Path("test"), bands_to_use=[-1], transform=None
                )
            with pytest.raises(ValueError, match=r"Bands have to be between 0 and \d+, but got -?\d+\."):
                ReferencePixels(
                    reference=pathlib.Path("test"),
                    annotated=pathlib.Path("test"),
                    bands_to_use=[0, 2, 8],
                    transform=None,
                )
            with pytest.raises(ValueError, match=r"Alpha channel have to be between -1 and \d+, but got -?\d+\."):
                ReferencePixels(
                    reference=pathlib.Path("test"),
                    annotated=pathlib.Path("test"),
                    bands_to_use=None,
                    alpha_channel=-2,
                    transform=None,
                )
            with pytest.raises(ValueError, match=r"Alpha channel have to be between -1 and \d+, but got -?\d+\."):
                ReferencePixels(
                    reference=pathlib.Path("test"),
                    annotated=pathlib.Path("test"),
                    bands_to_use=None,
                    alpha_channel=8,
                    transform=None,
                )
            # test black and white mask
            mp.setattr(ReferencePixels, "_load_mask", get_mock_load_mask(test_bw_mask))
            ReferencePixels(
                reference=pathlib.Path("test"), annotated=pathlib.Path("test"), bands_to_use=None, transform=None
            )
            # test mask of the wrong type
            mp.setattr(ReferencePixels, "_load_mask", get_mock_load_mask(test_wrong_size_mask))
            with pytest.raises(TypeError):
                ReferencePixels(
                    reference=pathlib.Path("test"), annotated=pathlib.Path("test"), bands_to_use=None, transform=None
                )
            # test mask which selects to few pixels
            mp.setattr(ReferencePixels, "_load_mask", get_mock_load_mask(test_too_small_mask))
            with pytest.raises(Exception, match=r"Not enough annotated pixels. Need at least \d+, but got \d+"):
                ReferencePixels(
                    reference=pathlib.Path("test"), annotated=pathlib.Path("test"), bands_to_use=None, transform=None
                )


class TestColorModels(unittest.TestCase):
    def setUp(self) -> None:
        self.monkeypatch = pytest.MonkeyPatch()

    def test_calculate_distance(self) -> None:
        def mock_reference_pixels_init(
            self: Any, bands_to_use: list[int], *args: Any, **kwargs: dict[str, Any]
        ) -> None:
            self.bands_to_use = bands_to_use
            self.values = test_reference_pixels_values
            self.transform = None

        with self.monkeypatch.context() as mp:
            mp.setattr(ReferencePixels, "__init__", mock_reference_pixels_init)
            # test Mahalanobis distance calculations
            md = MahalanobisDistance(bands_to_use=[0, 1, 2])
            np.testing.assert_almost_equal(md.calculate_distance(test_image), test_mahal_res, decimal=6)
            # test Gaussian Mixture Model distance calculations with 1 cluster
            gmmd1 = GaussianMixtureModelDistance(bands_to_use=[0, 1, 2], n_components=1)
            np.testing.assert_almost_equal(gmmd1.calculate_distance(test_image), test_gmm_1_res, decimal=6)
            # test Gaussian Mixture Model distance calculations with 2 cluster
            gmmd2 = GaussianMixtureModelDistance(bands_to_use=[0, 1, 2], n_components=2)
            np.testing.assert_almost_equal(gmmd2.calculate_distance(test_image), test_gmm_2_res, decimal=6)
