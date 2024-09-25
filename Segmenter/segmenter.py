import os
import time
import numpy as np
import rasterio
import multiprocessing
from datetime import datetime
import matplotlib.pyplot as plt

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
        self.mean_pixel_value=None
        self.image_statistics=[]
        self.number_of_tiles=None

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



    def calculate_statistics(self, tile_list):
        null_dist = self.colormodel.calculate_distance(np.ones((3, 1, 1))*255,self.transform)[0][0]
        self.number_of_tiles=len(tile_list)

        for tile in tile_list:
            tile_img=tile.read_tile()
            if not np.max(tile_img[:, :]) == np.min(tile_img[:, :]):
                image_statistics = np.histogram(tile_img, bins=256, range=(0, 255))[0]

                # Empty pixel are not counted in the histogram. 
                # Unwanted side effect is that pixels with a similar distance will also be discarded.
                image_statistics[int(null_dist)] = 0
                self.image_statistics=np.append(self.image_statistics,image_statistics)

        mean_divide = 0
        mean_sum = 0
        for x in range(0, 256):
            mean_sum += self.image_statistics[x] * x
            mean_divide += self.image_statistics[x]

        
        
        self.mean_pixel_value = mean_sum / mean_divide
        
        
    def save_statistics(self,args):
        statistics_path = self.output_tile_location + "/statistics"
        self.ensure_parent_directory_exist(statistics_path)

        print(f"Writing statistics to the folder \"{ statistics_path }\"")

        # Plot histogram of pixel valuess
        plt.plot(self.image_statistics)
        plt.title("Histogram of pixel values")
        plt.xlabel("Pixel Value")
        plt.ylabel("Number of Pixels")
        plt.savefig(statistics_path + "/Histogram of pixel values", dpi=300)
        plt.close()

        f = open(statistics_path + "/output_file.txt", "w")
        f.write("Input parameters:\n")
        f.write(f" - Orthomosaic: {args.orthomosaic}\n")
        f.write(f" - Reference image: {args.reference}\n")
        f.write(f" - Annotated image: {args.mask}\n")
        f.write(f" - Output scale factor: {args.scale}\n")
        f.write(f" - Tile sizes: {args.tile_size}\n")
        f.write(f" - Output tile location: {args.output_tile_location}\n")
        f.write(f" - Method: {args.method}\n")
        f.write(f" - Parameter: {args.param}\n")
        f.write(f" - Transform: {args.transform}\n")
        f.write(f" - reference pixels file: {args.ref_pixel_filename}\n")
        f.write(f" - Date and time of execution: {datetime.now().replace(microsecond=0)}\n")

        f.write("\n\nOutput from run\n")
        f.write(" - Average color value of annotated pixels\n")
        f.write(f" - {self.colormodel.average}\n")
        f.write(" - Covariance matrix of the annotated pixels\n")
        f.write(" - " + str(self.colormodel.covariance).replace('\n', '\n   ') + "\n")
        
        f.write(f" - Mean pixel value: {self.mean_pixel_value}\n")

        f.write(f" - Number of tiles: {self.number_of_tiles}\n")
        
        f.close()


