import numpy as np
import rasterio

from tile import Tile


class OrthomosaicTiles:
    def __init__(self, orthomosaic, tile_size, run_specific_tile, run_specific_tileset, **kwargs):
        self.orthomosaic = orthomosaic
        self.tile_size = tile_size
        self.overlap = 0.01
        self.run_specific_tile = run_specific_tile
        self.run_specific_tileset = run_specific_tileset

    def divide_orthomosaic_into_tiles(self):
        processing_tiles = self.get_processing_tiles()
        specified_processing_tiles = self.get_list_of_specified_tiles(processing_tiles)
        return specified_processing_tiles

    def get_list_of_specified_tiles(self, tile_list):
        specified_tiles = []
        if self.run_specific_tile is None and self.run_specific_tileset is None:
            return tile_list
        if self.run_specific_tile is not None:
            for tile_number in self.run_specific_tile:
                specified_tiles.append(tile_list[tile_number])
        if self.run_specific_tileset is not None:
            for start, end in zip(self.run_specific_tileset[::2], self.run_specific_tileset[1::2], strict=False):
                for tile_number in range(start, end + 1):
                    specified_tiles.append(tile_list[tile_number])
        return specified_tiles

    def get_processing_tiles(self):
        """
        Generate a list of tiles to process, including a padding region around
        the actual tile.
        Takes care of edge cases, where the tile does not have adjacent tiles in
        all directions.
        """
        processing_tiles, st_width, st_height = self.define_tiles()
        no_r = np.max([t.tile_position[0] for t in processing_tiles])
        no_c = np.max([t.tile_position[1] for t in processing_tiles])
        half_overlap_c = (self.tile_size - st_width) / 2
        half_overlap_r = (self.tile_size - st_height) / 2
        for tile_number, tile in enumerate(processing_tiles):
            tile.tile_number = tile_number
            tile.processing_range = [
                [half_overlap_r, self.tile_size - half_overlap_r],
                [half_overlap_c, self.tile_size - half_overlap_c],
            ]
            if tile.tile_position[0] == 0:
                tile.processing_range[0][0] = 0
            if tile.tile_position[0] == no_r:
                tile.processing_range[0][1] = self.tile_size
            if tile.tile_position[1] == 0:
                tile.processing_range[0][0] = 0
            if tile.tile_position[1] == no_c:
                tile.processing_range[0][1] = self.tile_size
        return processing_tiles

    def define_tiles(self):
        """
        Given a path to an orthomosaic, create a list of tiles which covers the
        orthomosaic with a specified overlap, height and width.
        """
        with rasterio.open(self.orthomosaic) as src:
            columns = src.width
            rows = src.height
            resolution = src.res
            crs = src.crs
            left = src.bounds[0]
            top = src.bounds[3]
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
