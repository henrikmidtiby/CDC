"""Tile Orthomosaics into smaller pieces for easier processing."""

from __future__ import annotations

import os
import pathlib
from typing import Any

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.transform import Affine
from rasterio.windows import Window


class Tile:
    """
    Handle all information of a tile with read and write.

    Parameters
    ----------
    start_point
    position
    height
    width
    resolution
    crs
    left
    top
    """

    def __init__(
        self,
        start_point: tuple[int, int],
        position: list[int],
        height: float,
        width: float,
        resolution: tuple[float, float],
        crs: rasterio.CRS,
        left: float,
        top: float,
    ):
        # Data for the tile
        self.size = (height, width)
        self.tile_position = position
        self.ulc = start_point
        self.lrc = (start_point[0] + height, start_point[1] + width)
        self.processing_range: list[list[float]] = [[0, 0], [0, 0]]
        self.resolution = resolution
        self.crs = crs
        self.left = left
        self.top = top
        self.ulc_global = [
            self.top - (self.ulc[0] * self.resolution[0]),
            self.left + (self.ulc[1] * self.resolution[1]),
        ]
        self.transform = Affine.translation(
            self.ulc_global[1] + self.resolution[0] / 2, self.ulc_global[0] - self.resolution[0] / 2
        ) * Affine.scale(self.resolution[0], -self.resolution[0])
        self.range: list[list[float]] | None = None
        self.tile_number: int = 0
        """The tile number."""
        self.output: NDArray[Any] = np.zeros(0)
        """np.ndarray : processed output of tile to save for later use."""

    def read_tile(self, orthomosaic_filename: pathlib.Path) -> NDArray[Any]:
        """Read the tiles image data from the orthomosaic."""
        with rasterio.open(orthomosaic_filename) as src:
            window = Window.from_slices(
                (self.ulc[0], self.lrc[0]),
                (self.ulc[1], self.lrc[1]),
            )
            img: NDArray[Any] = src.read(window=window)
            mask = src.read_masks(window=window)
            self.mask = mask[0]
            for band in range(mask.shape[0]):
                self.mask = self.mask & mask[band]
        return img

    def save_tile(self, image: NDArray[Any], output_tile_location: pathlib.Path) -> None:
        """Save the image of the tile to a tiff file. Filename is the tile number."""
        self.output = image
        if not output_tile_location.is_dir():
            os.makedirs(output_tile_location)
        output_tile_filename = output_tile_location.joinpath(f"{self.tile_number:05d}.tiff")
        with rasterio.open(
            output_tile_filename,
            "w",
            driver="GTiff",
            nodata=255,
            res=self.resolution,
            height=self.size[0],
            width=self.size[1],
            count=image.shape[0],
            dtype=image.dtype,
            crs=self.crs,
            transform=self.transform,
        ) as new_dataset:
            output = np.where(self.mask > 0, image, 255)
            new_dataset.write(output)
            new_dataset.write_mask(self.mask)


class OrthomosaicTiles:
    """
    Convert orthomosaic into tiles.

    Parameters
    ----------
    orthomosaic
    tile_size
        tile size in pixels.
    run_specific_tile
        List of tiles to run e.g. [15, 65] runs tiles 15 and 65.
    run_specific_tileset
        List of ranges of tiles to run e.g. [15, 65] runs all tiles between 15 and 65.
    """

    def __init__(
        self,
        *,
        orthomosaic: pathlib.Path,
        tile_size: int,
        run_specific_tile: list[int] | None = None,
        run_specific_tileset: list[int] | None = None,
    ):
        self.orthomosaic = orthomosaic
        self.tile_size = tile_size
        self.overlap = 0.01
        self.run_specific_tile = run_specific_tile
        self.run_specific_tileset = run_specific_tileset
        self.tiles: list[Tile] = []
        """List of tiles"""

    def divide_orthomosaic_into_tiles(self) -> list[Tile]:
        """Divide orthomosaic into tiles and select specific tiles if desired."""
        tiles = self.get_tiles()
        specified_tiles = self.get_list_of_specified_tiles(tiles)
        self.tiles = specified_tiles
        return specified_tiles

    def get_list_of_specified_tiles(self, tile_list: list[Tile]) -> list[Tile]:
        """From a list of all tiles select only specified tiles."""
        specified_tiles = []
        if self.run_specific_tile is None and self.run_specific_tileset is None:
            return tile_list
        if self.run_specific_tile is not None:
            for tile_number in self.run_specific_tile:
                specified_tiles.append(tile_list[tile_number])
        if self.run_specific_tileset is not None:
            for start, end in zip(self.run_specific_tileset[::2], self.run_specific_tileset[1::2], strict=True):
                if start > end:
                    raise ValueError(f"Specific tileset range is negative: from {start} to {end}")
                for tile_number in range(start, end + 1):
                    specified_tiles.append(tile_list[tile_number])
        return specified_tiles

    def get_tiles(self) -> list[Tile]:
        """
        Generate a list of tiles to process, including a padding region around
        the actual tile.
        Takes care of edge cases, where the tile does not have adjacent tiles in
        all directions.
        """
        tiles, st_width, st_height = self._define_tiles()
        no_r = np.max([t.tile_position[0] for t in tiles])
        no_c = np.max([t.tile_position[1] for t in tiles])
        half_overlap_c = (self.tile_size - st_width) / 2
        half_overlap_r = (self.tile_size - st_height) / 2
        for tile_number, tile in enumerate(tiles):
            tile.tile_number = tile_number
            tile.range = [
                [half_overlap_r, self.tile_size - half_overlap_r],
                [half_overlap_c, self.tile_size - half_overlap_c],
            ]
            if tile.tile_position[0] == 0:
                tile.range[0][0] = 0
            if tile.tile_position[0] == no_r:
                tile.range[0][1] = self.tile_size
            if tile.tile_position[1] == 0:
                tile.range[0][0] = 0
            if tile.tile_position[1] == no_c:
                tile.range[0][1] = self.tile_size
        return tiles

    def get_orthomosaic_data(self) -> tuple[int, int, tuple[float, float], rasterio.CRS, float, float]:
        """
        Read data from orthomosaic.

        Returns
        -------
        columns : int
        rows : int
        resolution : tuple[float, float]
        crs : rasterio.CRS
        left_corner : float
        top_corner : float
        """
        try:
            with rasterio.open(self.orthomosaic) as src:
                columns = src.width
                rows = src.height
                resolution = src.res
                crs = src.crs
                left = src.bounds[0]
                top = src.bounds[3]
        except rasterio.RasterioIOError as e:
            raise OSError(f"Could not open the orthomsaic at '{ self.orthomosaic }'") from e
        return columns, rows, resolution, crs, left, top

    def _define_tiles(self) -> tuple[list[Tile], int, int]:
        """
        Given a path to an orthomosaic, create a list of tiles which covers the
        orthomosaic with a specified overlap, height and width.

        Returns
        -------
        list of tiles : list[Tile]
        step width : int
        step height : int
        """
        columns, rows, resolution, crs, left, top = self.get_orthomosaic_data()
        last_position = (rows - self.tile_size, columns - self.tile_size)
        n_height = np.ceil(rows / (self.tile_size * (1 - self.overlap))).astype(int)
        n_width = np.ceil(columns / (self.tile_size * (1 - self.overlap))).astype(int)
        step_height = np.trunc(last_position[0] / (n_height - 1)).astype(int)
        step_width = np.trunc(last_position[1] / (n_width - 1)).astype(int)
        tiles = []
        for r in range(0, n_height):
            for c in range(0, n_width):
                pos = [r, c]
                if r == (n_height - 1):
                    tile_r = last_position[0]
                else:
                    tile_r = r * step_height
                if c == (n_width - 1):
                    tile_c = last_position[1]
                else:
                    tile_c = c * step_width
                tiles.append(Tile((tile_r, tile_c), pos, self.tile_size, self.tile_size, resolution, crs, left, top))
        return tiles, step_width, step_height

    def save_orthomosaic_from_tile_output(self, orthomosaic_filename: pathlib.Path) -> None:
        """Save an orthomosaic from the processed tiles."""
        with rasterio.open(self.orthomosaic) as src:
            profile = src.profile
            profile["count"] = 1
            profile["nodata"] = 255
        with rasterio.open(orthomosaic_filename, "w", **profile) as dst:
            for tile in self.tiles:
                window = Window.from_slices(
                    (tile.ulc[0], tile.lrc[0]),
                    (tile.ulc[1], tile.lrc[1]),
                )
                output = np.where(tile.mask > 0, tile.output, 255)
                dst.write(output, window=window)
                dst.write_mask(tile.mask, window=window)
