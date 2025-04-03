"""Tile Orthomosaics into smaller pieces for easier processing."""

from __future__ import annotations

import os
import pathlib
from typing import Any

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.enums import Resampling
from rasterio.windows import Window


class Tile:
    """
    Handle all information of a tile with read and write.

    Parameters
    ----------
    orthomosaic
        The orthomosaic from where the tile is taken.
    Upper_left_corner
        The pixel coordinate from the orthomosaic of the upper left corner of the tile in (columns, rows).
    position
        Tile position in orthomosaic in number of tile in (columns, rows).
    width
        Tile width.
    height
        Tile height.
    overlap
        Overlap as a fraction of width and height.
    number
        Used to identify tiles.
    """

    def __init__(
        self,
        orthomosaic: pathlib.Path,
        Upper_left_corner: tuple[int, int],
        position: tuple[int, int],
        width: float,
        height: float,
        overlap: float = 0.0,
        number: int = 0,
    ):
        # Data for the tile
        self.orthomosaic = orthomosaic
        self.size = (width, height)
        self.tile_position = position
        self.ulc = Upper_left_corner
        self.overlap = overlap
        self.tile_number = number
        """The tile number. Useful for identification."""
        self.output: NDArray[Any] = np.zeros(0)
        """np.ndarray : processed output of tile to save for later use."""
        windows = self.set_tile_data_from_orthomosaic()
        self.window = windows[0]
        """Window: Window specifying the region of the orthomosaic for this tile."""
        self.window_with_overlap = windows[1]
        """Window: Window specifying the region of the orthomosaic for this tile with overlap of neighboring tiles."""

    def set_tile_data_from_orthomosaic(self) -> tuple[Window, Window]:
        """Read data about the tile from the orthomosaic."""
        try:
            with rasterio.open(self.orthomosaic) as src:
                self.ortho_cols = src.width
                self.ortho_rows = src.height
                self.resolution = src.res
                self.crs = src.crs
                left = src.bounds[0]
                top = src.bounds[3]
                window_with_overlap = self._get_window(overlap=self.overlap)
                window = self._get_window(overlap=0)
                self.transform = src.window_transform(window_with_overlap)
        except rasterio.RasterioIOError as e:
            raise OSError(f"Could not open the orthomosaic at '{self.orthomosaic}'") from e
        self.ulc_global = [
            left + (self.ulc[0] * self.resolution[0]),
            top - (self.ulc[1] * self.resolution[1]),
        ]
        return window, window_with_overlap

    def _get_window(self, overlap: float) -> Window:
        pixel_overlap_width = int(self.size[0] * overlap)
        pixel_overlap_hight = int(self.size[1] * overlap)
        start_col = self.ulc[0] - pixel_overlap_width
        stop_col = self.ulc[0] + self.size[0] + pixel_overlap_width
        start_row = self.ulc[1] - pixel_overlap_hight
        stop_row = self.ulc[1] + self.size[1] + pixel_overlap_hight
        if start_col < 0:
            start_col = 0
        if stop_col > self.ortho_cols:
            stop_col = self.ortho_cols
        if start_row < 0:
            start_row = 0
        if stop_row > self.ortho_rows:
            stop_row = self.ortho_rows
        window = Window.from_slices(
            (start_row, stop_row),
            (start_col, stop_col),
        )
        return window

    def get_window_pixels_boundary(self) -> tuple[int, int, int, int]:
        """
        Get the tiles boundary without the overlap.

        Returns
        -------
        start_column : int
        stop_column : int
        start_row : int
        stop_row : int
        """
        c1 = self.window.col_off - self.window_with_overlap.col_off
        r1 = self.window.row_off - self.window_with_overlap.row_off
        c2 = c1 + self.window.width
        r2 = r1 + self.window.height
        return c1, c2, r1, r2

    def get_window_pixels(self, image: NDArray[Any]) -> NDArray[Any]:
        """Get pixels from tile without overlap."""
        c1, c2, r1, r2 = self.get_window_pixels_boundary()
        return image[:, r1:r2, c1:c2]

    def read_tile(self, with_overlap: bool = True) -> NDArray[Any]:
        """
        Read the tiles image data from the orthomosaic.
        If with_overlap is true a window with a border around the tile is used.
        """
        if with_overlap:
            window = self.window_with_overlap
        else:
            window = self.window
        with rasterio.open(self.orthomosaic) as src:
            img: NDArray[Any] = src.read(window=window)
            mask = src.read_masks(window=window)
            self.mask = mask[0]
            for band in range(mask.shape[0]):
                self.mask = self.mask & mask[band]
        return img, mask

    def save_tile(self, image: NDArray[Any], mask: NDArray[Any], output_tile_location: pathlib.Path) -> None:
        """Save the image of the tile to a tiff file. Filename is the tile number."""
        self.output = image
        if not output_tile_location.is_dir():
            os.makedirs(output_tile_location)
        output_tile_filename = output_tile_location.joinpath(f"{self.tile_number:05d}.tiff")
        with rasterio.open(
            output_tile_filename,
            "w",
            driver="GTiff",
            res=self.resolution,
            width=image.shape[2],
            height=image.shape[1],
            count=image.shape[0],
            dtype=image.dtype,
            crs=self.crs,
            transform=self.transform,
        ) as new_dataset:
            new_dataset.write(image)
            if image.shape[0] == 1:
                new_dataset.write_mask(mask)


class OrthomosaicTiles:
    """
    Convert orthomosaic into tiles.

    Parameters
    ----------
    orthomosaic
    tile_size
        tile size in pixels. Either a tuple with (width, height) or integer for square tiles.
    overlap
        How much the tiles should overlap as a fraction of the tile size.
    run_specific_tile
        List of tiles to run e.g. [15, 65] runs tiles 15 and 65.
    run_specific_tileset
        List of ranges of tiles to run e.g. [15, 65] runs all tiles between 15 and 65.
    """

    def __init__(
        self,
        *,
        orthomosaic: pathlib.Path,
        tile_size: int | tuple[int, int],
        overlap: float = 0,
        run_specific_tile: list[int] | None = None,
        run_specific_tileset: list[int] | None = None,
    ):
        self.orthomosaic = orthomosaic
        if type(tile_size) is tuple:
            self.tile_size = tile_size
        elif type(tile_size) is int:
            self.tile_size = (tile_size, tile_size)
        else:
            raise TypeError("Tile size must be int or tuple(int, int).")
        self.overlap = overlap
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

    def get_orthomosaic_size(self) -> tuple[int, int]:
        """
        Read size from orthomosaic.

        Returns
        -------
        columns : int
        rows : int
        """
        try:
            with rasterio.open(self.orthomosaic) as src:
                columns = src.width
                rows = src.height
        except rasterio.RasterioIOError as e:
            raise OSError(f"Could not open the orthomosaic at '{self.orthomosaic}'") from e
        return columns, rows

    def get_tiles(self) -> list[Tile]:
        """
        Given a path to an orthomosaic, create a list of tiles which covers the
        orthomosaic with a specified overlap, height and width.

        Returns
        -------
        list of tiles : list[Tile]
        """
        columns, rows = self.get_orthomosaic_size()
        n_width = np.ceil(columns / self.tile_size[0]).astype(int)
        n_height = np.ceil(rows / self.tile_size[1]).astype(int)
        tiles = []
        for r in range(0, n_height):
            for c in range(0, n_width):
                pos = (c, r)
                number = r * n_width + c
                tile_c = c * self.tile_size[0]
                tile_r = r * self.tile_size[1]
                tiles.append(
                    Tile(
                        self.orthomosaic,
                        (tile_c, tile_r),
                        pos,
                        self.tile_size[0],
                        self.tile_size[1],
                        self.overlap,
                        number,
                    )
                )
        return tiles

    def save_orthomosaic_from_tile_output(self, orthomosaic_filename: pathlib.Path) -> None:
        """Save an orthomosaic from the processed tiles."""
        if not orthomosaic_filename.parent.exists():
            orthomosaic_filename.parent.mkdir(parents=True)
        output_count = self.tiles[0].output.shape[0]
        with rasterio.open(self.orthomosaic) as src:
            profile = src.profile
            profile["count"] = output_count
            overview_factors = src.overviews(src.indexes[0])
        with rasterio.open(orthomosaic_filename, "w", **profile) as dst:
            for tile in self.tiles:
                output = tile.get_window_pixels(tile.output)
                dst.write(output, window=tile.window)
                if output_count == 1:
                    mask = tile.get_window_pixels(tile.mask)
                    dst.write_mask(mask, window=tile.window)
        with rasterio.open(orthomosaic_filename, "r+") as dst:
            dst.build_overviews(overview_factors, Resampling.average)
