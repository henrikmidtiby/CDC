import os
import time
import argparse
import numpy as np
from tqdm import tqdm


import colormodels
import tiler
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







def main(args):    
    referencepixels=colormodels.get_referencepixels(args.reference,args.mask,args.bands_to_use,args.ref_pixel_filename,args.ref_method)
    colormodel=colormodels.initialize_colormodel(referencepixels,args.method)
    cbs = ColorBasedSegmenter()
    cbs.initialize_segmenter(args.output_tile_location,colormodel,args.scale)
    if args.notiling==False:
        tile_list=tiler.get_tilelist(args.orthomosaic,int(args.tile_size))
           
        cbs.apply_colormodel_to_tiles(tile_list)
    if args.notiling==True:
        single_tile=tiler.get_single_tile(args.orthomosaic)
        cbs.apply_colormodel_to_single_tile(single_tile)
        

    

    













parser = argparse.ArgumentParser(
          prog='ColorDistranceCalculatorForOrthomosaics',
          description='A tool for calculating color distances in an '
                      'orthomosaic to a reference color based on samples from '
                      'an annotated image.',
          epilog='Program written by Henrik Skov Midtiby (hemi@mmmi.sdu.dk) in '
                 '2023 as part of the Precisionseedbreeding project supported '
                 'by GUDP and Frøafgiftsfonden.')
parser.add_argument('orthomosaic', 
                    help='Path to the orthomosaic that you want to process.')
parser.add_argument('reference', 
                    help='Path to the reference image.')
parser.add_argument('mask', 
                    help='Path to the annotated reference image.')
parser.add_argument('--bands_to_use',
                    default=None,
                    type=int,
                    nargs='+',
                    help='The bands needed to be analysed, written as a lis, 0 indexed'
                         'If no value is speciefied all bands except alpha channel will be analysed.')
parser.add_argument('--ref_pixel_filename',
                    default='Referencepixels.txt',)
parser.add_argument('--scale', 
                    default=5,
                    type=float,
                    help='The calculated distances are multiplied with this '
                         'factor before the result is saved as an image. '
                         'Default value is 5.')
parser.add_argument('--tile_size',
                    default=3000,
                    help='The height and width of tiles that are analyzed. '
                         'Default is 3000.')
parser.add_argument('--output_tile_location', 
                    default='output/mahal',
                    help='The location in which to save the mahalanobis tiles.')
parser.add_argument('--input_tile_location', 
                    default=None,
                    help='The location in which to save the input tiles.')
parser.add_argument('--method',
                    default='mahalanobis',
                    help='The method used for calculating distances from the '
                         'set of annotated pixels. '
                         'Possible values are \'mahalanobis\' for using the '
                         'Mahalanobis distance and '
                         '\'gmm\' for using a Gaussian Mixture Model.'
                         '\'mahalanobis\' is the default value.')
parser.add_argument('--param',
                    default=2,
                    type=int,
                    help='Numerical parameter for the color model. '
                         'When using the \'gmm\' method, this equals the '
                         'number of components in the Gaussian Mixture Model.')
parser.add_argument( '--notiling',
                    action="store_true",
                    help='Options for choosing not to seperate orthomosaic into tiles')
parser.add_argument( '--ref_method',
                    default=None,
                    help='Method for generating Reference pixels, default is from Mask( .tiff file), other option is red anotated jpg file.')
args = parser.parse_args()



#if args.method == 'gmm':
#   cbs.colormodel = GaussianMixtureModelDistance(args.param)


main(args)
