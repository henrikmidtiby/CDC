import pathlib
import unittest
from typing import Any

import pytest
from rasterio.transform import Affine

from CDC.orthomosaic_tiler import OrthomosaicTiles, Tile


class TestTiles(unittest.TestCase):
    def test_tile(self) -> None:
        tile_args = {
            "start_point": (1, 2),
            "position": [0, 0],
            "height": 300,
            "width": 400,
            "resolution": (0.5, 0.6),
            "crs": "test",
            "left": 100.2,
            "top": 100.3,
        }
        t_tile = Tile(**tile_args)  # type: ignore[arg-type]
        assert t_tile.ulc == (1, 2)
        assert t_tile.lrc == (1 + 300, 2 + 400)
        transform = Affine.translation(100.2 + (2 * 0.6) + 0.5 / 2, 100.3 - (1 * 0.5) - 0.5 / 2) * Affine.scale(
            0.5, -0.5
        )
        assert t_tile.transform == transform


class TestOrthomosaicTiler(unittest.TestCase):
    def setUp(self) -> None:
        self.monkeypatch = pytest.MonkeyPatch()

    def test_orthomosaic_tiler(self) -> None:
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

        orthomosaic_tiler_args = {
            "orthomosaic": pathlib.Path("/test/home/ortho.tiff"),
            "tile_size": 400,
            "run_specific_tile": [3, 50, 179],
            "run_specific_tileset": None,
        }
        with self.monkeypatch.context() as mp:
            mp.setattr(OrthomosaicTiles, "get_orthomosaic_data", mock_get_orthomosaic_data)
            ortho_tiler = OrthomosaicTiles(**orthomosaic_tiler_args)  # type: ignore[arg-type]
            tiles, _, _ = ortho_tiler._define_tiles()
            assert len(tiles) == int(8000 / 400 + 1) * int(4000 / 400 + 1)
            p_tiles = ortho_tiler.get_tiles()
            assert len(p_tiles) == int(8000 / 400 + 1) * int(4000 / 400 + 1)
            s_tiles = ortho_tiler.get_list_of_specified_tiles(p_tiles)
            assert len(s_tiles) == 3
            assert s_tiles[0].tile_number == 3
            assert s_tiles[1].tile_number == 50
            assert s_tiles[2].tile_number == 179
            ortho_tiler.run_specific_tile = None
            ortho_tiler.run_specific_tileset = [23, 56]
            s_tiles = ortho_tiler.get_list_of_specified_tiles(p_tiles)
            assert len(s_tiles) == 56 - 23 + 1
            assert s_tiles[0].tile_number == 23
            assert s_tiles[-1].tile_number == 56
            ortho_tiler.run_specific_tile = [5000]
            ortho_tiler.run_specific_tileset = None
            with pytest.raises(IndexError):
                ortho_tiler.get_list_of_specified_tiles(p_tiles)
            ortho_tiler.run_specific_tile = 5000  # type: ignore[assignment]
            with pytest.raises(TypeError):
                ortho_tiler.get_list_of_specified_tiles(p_tiles)
            ortho_tiler.run_specific_tile = None
            ortho_tiler.run_specific_tileset = [0, 5000]
            with pytest.raises(IndexError):
                ortho_tiler.get_list_of_specified_tiles(p_tiles)
            ortho_tiler.run_specific_tileset = [5, 3]
            with pytest.raises(ValueError, match=r"Specific tileset range is negative: from \d+ to \d+"):
                ortho_tiler.get_list_of_specified_tiles(p_tiles)
            ortho_tiler.run_specific_tileset = [3, 5, 7]
            with pytest.raises(ValueError, match=r"zip\(\) argument \d+ is shorter than argument \d+"):
                ortho_tiler.get_list_of_specified_tiles(p_tiles)
            ortho_tiler.run_specific_tile = None
            ortho_tiler.run_specific_tileset = None
            s_tiles = ortho_tiler.divide_orthomosaic_into_tiles()
            assert len(s_tiles) == int(8000 / 400 + 1) * int(4000 / 400 + 1)
