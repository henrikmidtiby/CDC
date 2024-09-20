
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
import os


def get_tilelist(orthomosaic,tile_size):

    orthomosaic_to_tiles=convert_orthomosaic_to_list_of_tiles()
    orthomosaic_to_tiles.tile_size= tile_size
    
    tile_list=orthomosaic_to_tiles.convert(orthomosaic)

    return tile_list

def get_single_tile(orthomosaic):
     tile=None
     with rasterio.open(orthomosaic) as src:
            columns = src.width
            rows = src.height
            resolution = src.res
            crs = src.crs
            left = src.bounds[0]
            top = src.bounds[3]
            tile=Tile((0,0),[0,0],rows,columns,resolution,crs,left,top,orthomosaic)
            im=src.read()
            tile.img=im
            tile.tile_number=0
     return tile 

class Tile:
    def __init__(self, start_point, position, height, width, 
                 resolution, crs, left, top,path_to_orthomosaic):
        # Data for the tile
        self.size = (height, width)
        self.tile_position = position
        self.ulc = start_point
        self.lrc = (start_point[0] + height, start_point[1] + width)
        self.processing_range = [[0, 0], [0, 0]]

        self.resolution = resolution
        self.crs = crs
        self.left = left
        self.top = top
        self.path_to_orthomosaic = path_to_orthomosaic
        self.ulc_global = [
                self.top - (self.ulc[0] * self.resolution[0]), 
                self.left + (self.ulc[1] * self.resolution[1])]
        self.transform = Affine.translation(
            self.ulc_global[1] + self.resolution[0] / 2,
            self.ulc_global[0] - self.resolution[0] / 2) * \
            Affine.scale(self.resolution[0], -self.resolution[0])

        self.tile_number = None

        self.img = None
    
    def ensure_parent_directory_exist(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

    def read_tile(self):
        with rasterio.open(self.path_to_orthomosaic) as src:
            window = Window.from_slices((self.ulc[0], self.lrc[0]),
                                        (self.ulc[1], self.lrc[1]))
            im = src.read(window=window)
        return im


    def save_tile(self,output_tile_location):
        if  output_tile_location is not None:
            self.ensure_parent_directory_exist(output_tile_location)
            name_mahal_results = f'{ output_tile_location }/mahal{ self.tile_number:04d}.tiff'
            img_to_save = self.img
            channels = img_to_save.shape[0]
            
            new_dataset = rasterio.open(name_mahal_results,
                                        'w',
                                        driver='GTiff',
                                        res=self.resolution,
                                        height=self.size[0],
                                        width=self.size[1],
                                        count=channels,
                                        dtype=img_to_save.dtype,
                                        crs=self.crs,
                                        transform=self.transform)
            new_dataset.write(img_to_save)
            new_dataset.close()

class convert_orthomosaic_to_list_of_tiles:
    def __init__(self):
        self.tile_size = 3000
        self.run_specific_tile = None
        self.run_specific_tileset = None

        self.resolution = None
        self.crs = None       
        self.left = None
        self.top = None

        self.filename_orthomosaic = None

    


    def convert(self, filename_orthomosaic):
        self.filename_orthomosaic = filename_orthomosaic
        self.divide_orthomosaic_into_tiles()
        return self.specified_tiles


    def divide_orthomosaic_into_tiles(self):
        with rasterio.open(self.filename_orthomosaic) as src:
            self.resolution = src.res
            self.crs = src.crs
            self.left = src.bounds[0]
            self.top = src.bounds[3]

        processing_tiles = self.get_processing_tiles(self.tile_size)

        self.specified_tiles = self.get_list_of_specified_tiles(processing_tiles)
        
        


    def get_list_of_specified_tiles(self, tile_list):
        specified_tiles = []
        for tile_number, tile in enumerate(tile_list):
            if self.run_specific_tileset is not None or self.run_specific_tile is not None:
                if self.run_specific_tileset is not None:
                    if tile_number >= self.run_specific_tileset[0] and tile_number <= self.run_specific_tileset[1]:
                        specified_tiles.append(tile)
                if self.run_specific_tile is not None:
                    if tile_number in self.run_specific_tile:
                        specified_tiles.append(tile)
            else:
                specified_tiles.append(tile)

        return specified_tiles


    def get_processing_tiles(self, tile_size):
        """
        Generate a list of tiles to process, including a padding region around
        the actual tile.
        Takes care of edge cases, where the tile does not have adjacent tiles in
        all directions.
        """
        processing_tiles, st_width, st_height = self.define_tiles(
            0.01, tile_size, tile_size)

        no_r = np.max([t.tile_position[0] for t in processing_tiles])
        no_c = np.max([t.tile_position[1] for t in processing_tiles])

        half_overlap_c = (tile_size-st_width)/2
        half_overlap_r = (tile_size-st_height)/2

        for tile_number, tile in enumerate(processing_tiles):
            tile.tile_number = tile_number
            tile.processing_range = [[half_overlap_r, tile_size - half_overlap_r],
                                     [half_overlap_c, tile_size - half_overlap_c]]
            if tile.tile_position[0] == 0:
                tile.processing_range[0][0] = 0
            if tile.tile_position[0] == no_r:
                tile.processing_range[0][1] = tile_size
            if tile.tile_position[1] == 0:
                tile.processing_range[0][0] = 0
            if tile.tile_position[1] == no_c:
                tile.processing_range[0][1] = tile_size

        return processing_tiles
    
    
    def define_tiles(self, overlap, height, width):
        """
        Given a path to an orthomosaic, create a list of tiles which covers the
        orthomosaic with a specified overlap, height and width.
        """

        with rasterio.open(self.filename_orthomosaic) as src:
            columns = src.width
            rows = src.height
            resolution = src.res
            crs = src.crs
            left = src.bounds[0]
            top = src.bounds[3]

        if rows < height and columns < width:
            raise Exception("tilesize larger than orthomosaic")
        else:
            last_position = (rows - height, columns - width)

            n_height = np.ceil(rows / (height * (1 - overlap))).astype(int)
            n_width = np.ceil(columns / (width * (1 - overlap))).astype(int)

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
                    tiles.append(Tile((tile_r, tile_c), pos, height, width, 
                                    resolution, crs, left, top,self.filename_orthomosaic))

            return tiles, step_width, step_height
