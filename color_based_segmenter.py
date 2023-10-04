"""
Copyright 2023 Henrik Skov Midtiby, hemi@mmmi.sdu.du, University of Southern Denmark
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
import time
import os
from icecream import ic
import argparse
from tqdm import tqdm
from sklearn import mixture


class Tile:
    def __init__(self, start_point, position, height, width):
        self.size = (height, width)
        self.tile_position = position
        self.ulc = start_point
        self.ulc_global = []
        self.lrc = (start_point[0] + height, start_point[1] + width)
        self.processing_range = [[0, 0], [0, 0]]


def rasterio_opencv2(image):
    if image.shape[0] >= 3:  # might include alpha channel
        false_color_img = image.transpose(1, 2, 0)
        separate_colors = cv2.split(false_color_img)
        return cv2.merge([separate_colors[2],
                          separate_colors[1],
                          separate_colors[0]])
    else:
        return image


def read_tile(orthomosaic, tile):
    with rasterio.open(orthomosaic) as src:
        window = Window.from_slices((tile.ulc[0], tile.lrc[0]),
                                    (tile.ulc[1], tile.lrc[1]))
        im = src.read(window=window)
    return rasterio_opencv2(im)

class colorspace:
    def __init__(self):
        self.colorspace = "bgr"

    def to_hsv(self, reference_img):
        return cv2.cvtColor(reference_img, cv2.COLOR_BGR2HSV)

    def to_lab(self, reference_img):
        return cv2.cvtColor(reference_img, cv2.COLOR_BGR2Lab)

    def convert_to_selected_colorspace(self, image):
        if self.colorspace == "bgr":
            return image
        elif self.colorspace == "hsv":
            return self.to_hsv(image)
        elif self.colorspace == "lab":
            return self.to_lab(image)
        else:
            raise Exception("Not a supported colorspace")

class ReferencePixels:
    def __init__(self):
        self.reference_image = None
        self.annotated_image = None
        self.pixel_mask = None
        self.values = None
        self.colorspace = colorspace()

    def load_reference_image(self, filename_reference_image):
        self.reference_image = self.colorspace.convert_to_selected_colorspace(cv2.imread(filename_reference_image))
        
    def load_annotated_image(self, filename_annotated_image):
        self.annotated_image = cv2.imread(filename_annotated_image)
        
    def generate_pixel_mask(self,
                            lower_range=(0, 0, 245),
                            higher_range=(10, 10, 256)):
        self.pixel_mask = cv2.inRange(self.annotated_image,
                                      lower_range,
                                      higher_range)
        pixels = np.reshape(self.reference_image, (-1, 3))
        mask_pixels = np.reshape(self.pixel_mask, (-1))
        self.values = pixels[mask_pixels == 255, ].transpose()

    def show_statistics_of_pixel_mask(self):
        print(f"Number of annotated pixels: { self.values.shape }")
        if self.values.shape[1] < 100:
            raise Exception("Not enough annotated pixels")

    def save_pixel_values_to_file(self, filename):
        print(f"Writing pixel values to the file \"{ filename }\"")
        np.savetxt(filename, 
                   self.values.transpose(), 
                   delimiter = '\t', 
                   fmt='%i', 
                   header = self.colorspace.colorspace[0] + "\t" 
                   + self.colorspace.colorspace[1] + "\t" 
                   + self.colorspace.colorspace[2],
                   comments = "")
        
        


class MahalanobisDistance:
    """
    A multivariate normal distribution used to describe the color of a set of
    pixels.
    """
    def __init__(self):
        self.average = None
        self.covariance = None

    def calculate_statistics(self, reference_pixels):
        self.covariance = np.cov(reference_pixels)
        self.average = np.average(reference_pixels, axis=1)

    def calculate_distance(self, image):
        """
        For all pixels in the image, calculate the Mahalanobis distance
        to the reference color.
        """
        pixels = np.reshape(image, (-1, 3))
        inv_cov = np.linalg.inv(self.covariance)
        diff = pixels - self.average
        modified_dot_product = diff * (diff @ inv_cov)
        distance = np.sum(modified_dot_product, axis=1)
        distance = np.sqrt(distance)

        distance_image = np.reshape(distance, (image.shape[0], image.shape[1]))

        return distance_image

    def show_statistics(self):
        print("Average color value of annotated pixels")
        print(self.average)
        print("Covariance matrix of the annotated pixels")
        print(self.covariance)


class GaussianMixtureModelDistance:
    def __init__(self, n_components):
        self.gmm = None
        self.n_components = n_components

    def calculate_statistics(self, reference_pixels):
        self.gmm = mixture.GaussianMixture(n_components=self.n_components,
                                           covariance_type="full")
        self.gmm.fit(reference_pixels.transpose())

    def calculate_distance(self, image):
        """
        For all pixels in the image, calculate the distance to the
        reference color modelled as a Gaussian Mixture Model.
        """
        pixels = np.reshape(image, (-1, 3))
        distance = self.gmm.score_samples(pixels)
        distance_image = np.reshape(distance, (image.shape[0], image.shape[1]))
        return distance_image

    def show_statistics(self):
        print("GMM")
        print(self.gmm)
        print(self.gmm.means_)
        print(self.gmm.covariances_)


class ColorBasedSegmenter:
    def __init__(self):
        self.top = None
        self.left = None
        self.crs = None
        self.resolution = None
        self.tile_size = 3000
        self.reference_pixels = ReferencePixels()
        self.colormodel = MahalanobisDistance()
        self.ref_image_filename = None
        self.ref_image_annotated_filename = None
        self.output_scale_factor = None
        self.output_tile_location = None
        self.input_tile_location = None
        self.process_tiles = 0
        self.pixel_mask_file = "pixel_values"

    def main(self, filename_orthomosaic):
        output_directory = os.path.dirname(self.output_tile_location)
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
        self.initialize_color_model(self.ref_image_filename,
                                    self.ref_image_annotated_filename)
        if self.process_tiles:
            self.process_orthomosaic(filename_orthomosaic)

    def initialize_color_model(self,
                               ref_image_filename,
                               ref_image_annotated_filename):
        self.reference_pixels.load_reference_image(ref_image_filename)
        self.reference_pixels.load_annotated_image(ref_image_annotated_filename)
        self.reference_pixels.generate_pixel_mask()
        self.reference_pixels.show_statistics_of_pixel_mask()
        self.reference_pixels.save_pixel_values_to_file(self.pixel_mask_file + ".csv")
        self.colormodel.calculate_statistics(self.reference_pixels.values)
        self.colormodel.show_statistics()

    def process_orthomosaic(self, filename_orthomosaic):
        start_time = time.time()
        self.calculate_color_distances_in_orthomosaic(filename_orthomosaic)
        proc_time = time.time() - start_time
        # print('Calculation of color distances: ', proc_time)

    def define_tiles(self, filename_orthomosaic, overlap, height, width):
        """
        Given a path to an orthomosaic, create a list of tiles which covers the
        orthomosaic with a specified overlap, height and width.
        """

        with rasterio.open(filename_orthomosaic) as src:
            columns = src.width
            rows = src.height

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
                tiles.append(Tile((tile_r, tile_c), pos, height, width))

        return tiles, step_width, step_height

    def calculate_color_distances_in_orthomosaic(self, filename_orthomosaic):
        """
        For all pixels in the orthomosaic, calculate the Mahalanobis distance 
        to the reference color.
        """
        with rasterio.open(filename_orthomosaic) as src:
            self.resolution = src.res
            self.crs = src.crs
            self.left = src.bounds[0]
            self.top = src.bounds[3]

        processing_tiles = self.get_processing_tiles(filename_orthomosaic,
                                                     self.tile_size)

        for tile_number, tile in enumerate(tqdm(processing_tiles)):
            img = self.reference_pixels.colorspace.convert_to_selected_colorspace(read_tile(filename_orthomosaic, tile))
            if self.is_image_empty(img):
                continue

            self.process_tile(filename_orthomosaic, img, tile_number, tile)

    def get_processing_tiles(self, filename_orthomosaic, tile_size):
        """
        Generate a list of tiles to process, including a padding region around
        the actual tile.
        Takes care of edge cases, where the tile does not have adjacent tiles in
        all directions.
        """
        processing_tiles, st_width, st_height = self.define_tiles(
            filename_orthomosaic, 0.01, tile_size, tile_size)

        no_r = np.max([t.tile_position[0] for t in processing_tiles])
        no_c = np.max([t.tile_position[1] for t in processing_tiles])

        half_overlap_c = (tile_size-st_width)/2
        half_overlap_r = (tile_size-st_height)/2

        for tile in processing_tiles:
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

    def is_image_empty(self, image):
        """Helper function for deciding if an image contains no data."""
        return np.max(image[:, :, 0]) == np.min(image[:, :, 0])

    def process_tile(self, filename_orthomosaic, img_rgb, tile_number, tile):
        tile.ulc_global = [
                self.top - (tile.ulc[0] * self.resolution[0]), 
                self.left + (tile.ulc[1] * self.resolution[1])]

        distance_image = self.colormodel.calculate_distance(img_rgb[:, :, :])
        distance = cv2.convertScaleAbs(distance_image,
                                       alpha=self.output_scale_factor,
                                       beta=0)
        distance = distance.astype(np.uint8)

        width = tile.size[1]
        height = tile.size[0]

        # TODO: Check that a proper coordinate system has been used for the original orthomosaic (ie. UTM).
        # Check the unit size of the coordinates of the upper left corners.
        transform = Affine.translation(
            tile.ulc_global[1] + self.resolution[0] / 2,
            tile.ulc_global[0] - self.resolution[0] / 2) * \
            Affine.scale(self.resolution[0], -self.resolution[0])

        # optional save of results - just lob detection and thresholding result
        self.save_results(img_rgb, tile_number, distance,
                          self.resolution, height, width, self.crs, transform)

    def save_results(self, img_rgb, tile_number, mahal,
                     res, height, width, crs, transform):
        if self.input_tile_location is not None:
            name_annotated_image = f'{ self.input_tile_location }{ tile_number:04d}.tiff'
            img_to_save = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            temp_to_save = img_to_save.transpose(2, 0, 1)
            new_dataset = rasterio.open(name_annotated_image,
                                        'w',
                                        driver='GTiff',
                                        res=res,
                                        height=height,
                                        width=width,
                                        count=3,
                                        dtype=temp_to_save.dtype,
                                        crs=crs,
                                        transform=transform)
            new_dataset.write(temp_to_save)
            new_dataset.close()

        if self.output_tile_location is not None:
            name_mahal_results = f'{ self.output_tile_location }{ tile_number:04d}.tiff'
            img_to_save = mahal
            temp_to_save = img_to_save.reshape(1, img_to_save.shape[0],
                                               img_to_save.shape[1])
            # The ordering should be color channels, width and height.
            new_dataset = rasterio.open(name_mahal_results,
                                        'w',
                                        driver='GTiff',
                                        res=res,
                                        height=height,
                                        width=width,
                                        count=1,
                                        dtype=temp_to_save.dtype,
                                        crs=crs,
                                        transform=transform)
            new_dataset.write(temp_to_save)
            new_dataset.close()


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
parser.add_argument('annotated', 
                    help='Path to the annotated reference image.')
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
parser.add_argument('--colorspace',
                    default='bgr',
                    help='Defines which colorspace will be used to find specific '
                         'colors in an orthomosaic. \n'
                         'Default is bgr(rgb), but cielab can be choosen with lab '
                         'and HSV can be choosen with hsv')
parser.add_argument('--process_tiles',
                    default=0,
                    help='Defaults to 0 which means that the tiles are not '
                         'processed, but a .csv file with the mask pixels is '
                         'returned')
parser.add_argument('--mask_file_name',
                    default= "pixel_values",
                    help='Change the name in which the pixel mask is saved. It '
                         'defaults to pixel_values (.csv is automatically added)')
args = parser.parse_args()


cbs = ColorBasedSegmenter()
ic(args.method)
if args.method == 'gmm':
    cbs.colormodel = GaussianMixtureModelDistance(args.param)
cbs.ref_image_filename = args.reference
cbs.ref_image_annotated_filename = args.annotated
cbs.output_scale_factor = args.scale
cbs.tile_size = args.tile_size
cbs.output_tile_location = args.output_tile_location
cbs.input_tile_location = args.input_tile_location
cbs.reference_pixels.colorspace.colorspace = args.colorspace
cbs.process_tiles = args.process_tiles
cbs.pixel_mask_file = args.mask_file_name 
cbs.main(args.orthomosaic)
