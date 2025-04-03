import pathlib
import unittest
from typing import Any

import numpy as np
import pytest
from numpy.random import default_rng
from numpy.typing import NDArray
from rasterio.windows import Window

from CDC.orthomosaic_tiler import OrthomosaicTiles, Tile
from CDC.tiled_color_based_distance import TiledColorBasedDistance

test_float_image_0_1 = default_rng(1234).random((3, 5, 5))
test_float_image_neg1_1 = test_float_image_0_1 * 2 - 1
test_uint8_image = (test_float_image_0_1 * 255).astype(np.uint8)

test_float_image_0_1_csa = np.minimum(np.abs(5 * test_float_image_0_1), 255)
test_float_image_neg1_1_csa = np.minimum(np.abs(5 * test_float_image_neg1_1), 255)
test_uint8_image_csa = np.minimum(np.abs(5 * test_uint8_image), 255)


class ColorModel:
    def calculate_distance(self, image: NDArray[Any]) -> NDArray[Any]:
        return image


class TestTiledColorSegmenter(unittest.TestCase):
    def setUp(self) -> None:
        self.monkeypatch = pytest.MonkeyPatch()

    def test_tiled_color_segmenter(self) -> None:
        # test convertScaleAbs
        np.testing.assert_equal(
            TiledColorBasedDistance.convertScaleAbs(test_float_image_0_1, 5), test_float_image_0_1_csa
        )
        np.testing.assert_equal(
            TiledColorBasedDistance.convertScaleAbs(test_float_image_neg1_1, 5), test_float_image_neg1_1_csa
        )
        np.testing.assert_equal(TiledColorBasedDistance.convertScaleAbs(test_uint8_image, 5), test_uint8_image_csa)

        def mock_set_tile_data_from_orthomosaic(
            self: Any, *args: Any, **kwargs: dict[str, Any]
        ) -> tuple[Window, Window]:
            self.ortho_cols = 4000
            self.ortho_rows = 3000
            self.resolution = (0.1, 0.1)
            self.crs = "test"
            left = 8000
            top = 6000
            window_with_overlap = self._get_window(overlap=self.overlap)
            window = self._get_window(overlap=0)
            self.transform = None
            self.ulc_global = [
                left + (self.ulc[0] * self.resolution[0]),
                top - (self.ulc[1] * self.resolution[1]),
            ]
            return window, window_with_overlap

        def mock_get_orthomosaic_size(*args: Any, **kwargs: dict[str, Any]) -> tuple[int, int]:
            columns = 8000
            rows = 4000
            return columns, rows

        def mock_read_tile(self: Any, *args: Any, **kwargs: dict[str, Any]) -> NDArray[Any]:
            return test_uint8_image, np.ones((1, *test_uint8_image.shape[1:]))

        ortho_tiler_args = {
            "orthomosaic": pathlib.Path("/test/home/ortho.tiff"),
            "tile_size": 400,
            "run_specific_tile": None,
            "run_specific_tileset": None,
        }

        tcbd_args = {
            "color_model": ColorModel(),
            "scale": 5,
            "output_location": pathlib.Path("/test/home/output"),
        }
        with self.monkeypatch.context() as mp:
            mp.setattr(Tile, "set_tile_data_from_orthomosaic", mock_set_tile_data_from_orthomosaic)
            mp.setattr(OrthomosaicTiles, "get_orthomosaic_size", mock_get_orthomosaic_size)
            mp.setattr(Tile, "read_tile", mock_read_tile)
            ortho_tiler = OrthomosaicTiles(**ortho_tiler_args)  # type: ignore[arg-type]
            tcbs = TiledColorBasedDistance(ortho_tiler=ortho_tiler, **tcbd_args)  # type: ignore[arg-type]
            np.testing.assert_equal(tcbs.process_image(test_float_image_0_1), test_float_image_0_1_csa.astype(np.uint8))
            np.testing.assert_equal(
                tcbs.process_image(test_float_image_neg1_1), test_float_image_neg1_1_csa.astype(np.uint8)
            )
            np.testing.assert_equal(tcbs.process_image(test_uint8_image), test_uint8_image_csa)
            assert len(tcbs.ortho_tiler.tiles) == int(8000 / 400) * int(4000 / 400)
