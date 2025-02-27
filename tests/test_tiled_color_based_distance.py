import pathlib
import unittest
from typing import Any

import numpy as np
import pytest
from numpy.random import default_rng
from numpy.typing import NDArray

from CDC.orthomosaic_tiler import OrthomosaicTiles, Tile
from CDC.tiled_color_based_distance import TiledColorBasedDistance

test_float_image_0_1 = default_rng(1234).random((3, 5, 5))
test_float_image_neg1_1 = test_float_image_0_1 * 2 - 1
test_uint8_image = (test_float_image_0_1 * 255).astype(np.uint8)

test_float_image_0_1_csa = np.minimum(np.abs(5 * test_float_image_0_1), 255)
test_float_image_neg1_1_csa = np.minimum(np.abs(5 * test_float_image_neg1_1), 255)
test_uint8_image_csa = np.minimum(np.abs(5 * test_uint8_image), 255)


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

        class ColorModel:
            def calculate_distance(self, image: NDArray[Any]) -> NDArray[Any]:
                return image

        def mock_get_orthomosaic_data(
            *args: Any, **kwargs: dict[str, Any]
        ) -> tuple[int, int, tuple[float, float], str, float, float]:
            columns = 8000
            rows = 4000
            resolution = (0.05, 0.05)
            crs = "test"
            left = 300000.0
            top = 6000000.0
            return columns, rows, resolution, crs, left, top

        def mock_read_tile(self: Any, *args: Any, **kwargs: dict[str, Any]) -> NDArray[Any]:
            self.mask = np.ones((1, *test_uint8_image.shape[1:]))
            return test_uint8_image

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
            mp.setattr(OrthomosaicTiles, "get_orthomosaic_data", mock_get_orthomosaic_data)
            mp.setattr(Tile, "read_tile", mock_read_tile)
            ortho_tiler = OrthomosaicTiles(**ortho_tiler_args)  # type: ignore[arg-type]
            tcbs = TiledColorBasedDistance(ortho_tiler=ortho_tiler, **tcbd_args)  # type: ignore[arg-type]
            np.testing.assert_equal(tcbs.process_image(test_float_image_0_1), test_float_image_0_1_csa.astype(np.uint8))
            np.testing.assert_equal(
                tcbs.process_image(test_float_image_neg1_1), test_float_image_neg1_1_csa.astype(np.uint8)
            )
            np.testing.assert_equal(tcbs.process_image(test_uint8_image), test_uint8_image_csa)
            tcbs.process_tiles(save_tiles=False, save_ortho=False)
            assert len(tcbs.ortho_tiler.tiles) == int(8000 / 400 + 1) * int(4000 / 400 + 1)
            np.testing.assert_equal(tcbs.ortho_tiler.tiles[0].output, test_uint8_image_csa)
            _, mean_pixel_value = tcbs._calculate_statistics()
            np.testing.assert_almost_equal(mean_pixel_value, np.float64(113.293333), decimal=6)
