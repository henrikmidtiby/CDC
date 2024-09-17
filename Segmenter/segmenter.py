import os
import time
import argparse
import numpy as np
from tqdm import tqdm



#git test

def rasterio_open(image):  # might include alpha channel
        # image has the shape ncolorchannels x height x width
        false_color_img = image.transpose(1, 2, 0)
        # false_color_image has the shape height x width x ncolorchannels
        return false_color_img

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
        output_directory = os.path.dirname(self.output_tile_location)
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
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
        output_directory = os.path.dirname(self.output_tile_location)
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
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
        if not self.is_image_empty(tile.img):
            distance_image = self.colormodel.calculate_distance(tile.img[:, :, :])
            distance =convertScaleAbs(distance_image,alpha=self.output_scale_factor)
            distance = distance.astype(np.uint8)
            tile.img = distance
            tile.save_tile(self.output_tile_location)

