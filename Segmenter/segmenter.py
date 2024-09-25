import os
import time
import numpy as np
import rasterio
import multiprocessing

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
        self.transform=None

    def apply_colormodel_to_tiles(self, tile_list):
        self.ensure_parent_directory_exist(self.output_tile_location)
        num_cores=multiprocessing.cpu_count()
        start = time.time()
       
        with multiprocessing.Pool(processes=num_cores) as pool:
        # Distribute the process_tiles function to all tiles
            list(tqdm(pool.imap(self.process_tile, tile_list), total=len(tile_list)))
        print("Time to run all tiles: ", time.time() - start)
    
   # def apply_threshold_to_image(threshhold,img):
    def apply_colormodel_to_single_tile(self, tile):
        self.ensure_parent_directory_exist(self.output_tile_location)
        start = time.time()
    
        self.process_tile(tile)
        print("Time to run all tiles: ", time.time() - start)     
    

    def initialize_segmenter(self,output_tile_location,colormodel,scalefactor=None,transform=None):
        self.output_tile_location=output_tile_location
        self.colormodel=colormodel
        if scalefactor is not None:
            self.output_scale_factor=scalefactor
        if transform is not None:
            self.transform=transform

    


    def is_image_empty(self, image):
        """Helper function for deciding if an image contains no data."""
        return np.max(image[:, :, 0]) == np.min(image[:, :, 0])
    
    def ensure_parent_directory_exist(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)


    def process_tile(self, tile):
        tile_img=tile.read_tile()
        if self.is_image_empty(tile_img):
                return
        if not self.is_image_empty(tile_img):
            distance_image = self.colormodel.calculate_distance(tile_img,self.transform)
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

