import pathlib
import unittest
from typing import Any

import numpy as np
import pytest
from rasterio.windows import Window

from CDC.orthomosaic_tiler import OrthomosaicTiles, Tile


def mock_set_tile_data_from_orthomosaic(self: Any, *args: Any, **kwargs: dict[str, Any]) -> tuple[Window, Window]:
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


class TestTiles(unittest.TestCase):
    def setUp(self) -> None:
        self.monkeypatch = pytest.MonkeyPatch()

    def test_tile(self) -> None:
        tile_args1 = {
            "orthomosaic": pathlib.Path("/test/home/ortho.tif"),
            "Upper_left_corner": (1, 2),
            "position": (0, 0),
            "width": 400,
            "height": 300,
            "overlap": 0.2,
        }
        tile_args2 = {
            "orthomosaic": pathlib.Path("/test/home/ortho.tif"),
            "Upper_left_corner": (400, 300),
            "position": (0, 0),
            "width": 400,
            "height": 300,
            "overlap": 0.2,
        }
        test_image = np.mgrid[0:1000, 0:1000]
        with self.monkeypatch.context() as mp:
            mp.setattr(Tile, "set_tile_data_from_orthomosaic", mock_set_tile_data_from_orthomosaic)
            t_tile = Tile(**tile_args1)  # type: ignore[arg-type]
            assert t_tile.ulc == (1, 2)
            assert t_tile.window == Window(1, 2, 400, 300)
            assert t_tile.window_with_overlap == Window(0, 0, 1 + 400 + int(400 * 0.2), 2 + 300 + int(300 * 0.2))
            assert t_tile.get_window_pixels_boundary() == (1, 1 + 400, 2, 2 + 300)
            np.testing.assert_equal(t_tile.get_window_pixels(test_image), np.mgrid[2:302, 1:401])
            t_tile = Tile(**tile_args2)  # type: ignore[arg-type]
            assert t_tile.ulc == (400, 300)
            assert t_tile.window == Window(400, 300, 400, 300)
            assert t_tile.window_with_overlap == Window(
                400 - int(400 * 0.2), 300 - int(300 * 0.2), 400 + 2 * int(400 * 0.2), 300 + 2 * int(300 * 0.2)
            )
            assert t_tile.get_window_pixels_boundary() == (
                int(400 * 0.2),
                400 + int(400 * 0.2),
                int(300 * 0.2),
                300 + int(300 * 0.2),
            )
            np.testing.assert_equal(t_tile.get_window_pixels(test_image), np.mgrid[60:360, 80:480])


class TestOrthomosaicTiler(unittest.TestCase):
    def setUp(self) -> None:
        self.monkeypatch = pytest.MonkeyPatch()

    def test_orthomosaic_tiler(self) -> None:
        orthomosaic_tiler_args1 = {
            "orthomosaic": pathlib.Path("/test/home/ortho.tiff"),
            "tile_size": 400,
            "overlap": 0.2,
            "run_specific_tile": [3, 50, 179],
            "run_specific_tileset": None,
        }
        orthomosaic_tiler_args2 = {
            "orthomosaic": pathlib.Path("/test/home/ortho.tiff"),
            "tile_size": (400, 300),
        }
        orthomosaic_tiler_args3 = {
            "orthomosaic": pathlib.Path("/test/home/ortho.tiff"),
            "tile_size": None,
        }
        with self.monkeypatch.context() as mp:
            mp.setattr(OrthomosaicTiles, "get_orthomosaic_size", mock_get_orthomosaic_size)
            mp.setattr(Tile, "set_tile_data_from_orthomosaic", mock_set_tile_data_from_orthomosaic)
            ortho_tiler = OrthomosaicTiles(**orthomosaic_tiler_args1)  # type: ignore[arg-type]
            tiles = ortho_tiler.get_tiles()
            assert len(tiles) == int(8000 / 400) * int(4000 / 400)
            s_tiles = ortho_tiler.get_list_of_specified_tiles(tiles)
            assert len(s_tiles) == 3
            assert s_tiles[0].tile_number == 3
            assert s_tiles[1].tile_number == 50
            assert s_tiles[2].tile_number == 179
            ortho_tiler.run_specific_tile = None
            ortho_tiler.run_specific_tileset = [23, 56]
            s_tiles = ortho_tiler.get_list_of_specified_tiles(tiles)
            assert len(s_tiles) == 56 - 23 + 1
            assert s_tiles[0].tile_number == 23
            assert s_tiles[-1].tile_number == 56
            ortho_tiler.run_specific_tile = [5000]
            ortho_tiler.run_specific_tileset = None
            with pytest.raises(IndexError):
                ortho_tiler.get_list_of_specified_tiles(tiles)
            ortho_tiler.run_specific_tile = 5000  # type: ignore[assignment]
            with pytest.raises(TypeError):
                ortho_tiler.get_list_of_specified_tiles(tiles)
            ortho_tiler.run_specific_tile = None
            ortho_tiler.run_specific_tileset = [0, 5000]
            with pytest.raises(IndexError):
                ortho_tiler.get_list_of_specified_tiles(tiles)
            ortho_tiler.run_specific_tileset = [5, 3]
            with pytest.raises(ValueError, match=r"Specific tileset range is negative: from \d+ to \d+"):
                ortho_tiler.get_list_of_specified_tiles(tiles)
            ortho_tiler.run_specific_tileset = [3, 5, 7]
            with pytest.raises(ValueError, match=r"zip\(\) argument \d+ is shorter than argument \d+"):
                ortho_tiler.get_list_of_specified_tiles(tiles)
            ortho_tiler.run_specific_tile = None
            ortho_tiler.run_specific_tileset = None
            s_tiles = ortho_tiler.divide_orthomosaic_into_tiles()
            assert len(s_tiles) == int(8000 / 400) * int(4000 / 400)
            ortho_tiler = OrthomosaicTiles(**orthomosaic_tiler_args2)  # type: ignore[arg-type]
            tiles = ortho_tiler.get_tiles()
            assert len(tiles) == int(8000 / 400) * int(np.ceil(4000 / 300))
            with pytest.raises(TypeError):
                ortho_tiler = OrthomosaicTiles(**orthomosaic_tiler_args3)  # type: ignore[arg-type]
