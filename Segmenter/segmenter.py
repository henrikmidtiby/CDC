import os
import time
import numpy as np
import rasterio

from tqdm import tqdm


def convertScaleAbs(img,alpha):
    scaled_img=alpha*img
    for i, value in np.ndenumerate(scaled_img):
        scaled_img[i]=min(value,255)
    return scaled_img


class ColorBasedSegmenter:
    def __init__(self):
        self.output_tile_location = None
        self.colormodel = None
        self.output_scale_factor = 5

    def apply_colormodel_to_tiles(self, tile_list):
        self.ensure_parent_directory_exist(self.output_tile_location)
        start = time.time()
        for tile in tqdm(tile_list):

            #img = self.reference_pixels.colorspace.convert_to_selected_colorspace(tile.img)
            if self.is_image_empty(tile.img):
                continue
            self.process_tile(tile)
        print("Time to run all tiles: ", time.time() - start)
    
   # def apply_threshold_to_image(threshhold,img):
    def apply_colormodel_to_single_tile(self, tile):
        self.ensure_parent_directory_exist(self.output_tile_location)
        start = time.time()
    
        self.process_tile(tile)
        print("Time to run all tiles: ", time.time() - start)     
    

    def initialize_segmenter(self,output_tile_location,colormodel,scalefactor=None):
        self.output_tile_location=output_tile_location
        self.colormodel=colormodel
        if scalefactor is not None:
            self.output_scale_factor=scalefactor


    def is_image_empty(self, image):
        """Helper function for deciding if an image contains no data."""
        return np.max(image[:, :, 0]) == np.min(image[:, :, 0])
    
    def ensure_parent_directory_exist(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)


    def process_tile(self, tile):
        tile_img=tile.read_tile()
        if not self.is_image_empty(tile_img):
            distance_image = self.colormodel.calculate_distance(tile_img)
            distance =convertScaleAbs(distance_image,alpha=self.output_scale_factor)
            distance = distance.astype(np.uint8)
            self.save_tile(distance,tile,self.output_tile_location)

    def save_tile(self,img,tile,output_tile_location):
        if  output_tile_location is not None:
            self.ensure_parent_directory_exist(output_tile_location)
            name_mahal_results = f'{ output_tile_location }/distance_tiles{ tile.tile_number:04d}.tiff'
            img_to_save = img
            channels = img_to_save.shape[0]
            new_dataset = rasterio.open(name_mahal_results,
                                        'w',
                                        driver='GTiff',
                                        res=tile.resolution,
                                        height=tile.size[0],
                                        width=tile.size[1],
                                        count=channels,
                                        dtype=img_to_save.dtype,
                                        crs=tile.crs,
                                        transform=tile.transform)
            new_dataset.write(img_to_save)
            new_dataset.close()

