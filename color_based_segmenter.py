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

import argparse
import os
import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
from tqdm import tqdm

from color_models import GaussianMixtureModelDistance, MahalanobisDistance
from convert_orthomosaic_to_list_of_tiles import convert_orthomosaic_to_list_of_tiles


class ColorSpace:
    def __init__(self, color_space=None):
        self.color_space = color_space if color_space is not None else "bgr"

    def convert_to_selected_color_space(self, image):
        if self.color_space == "bgr":
            return image
        elif self.color_space == "hsv":
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.color_space == "lab":
            return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        else:
            raise Exception("Not a supported colorspace")


class ReferencePixels:
    def __init__(self, color_space):
        self.reference_image = None
        self.annotated_image = None
        self.pixel_mask = None
        self.values = None
        self.color_space = color_space

    def load_reference_image(self, filename_reference_image):
        self.reference_image = self.color_space.convert_to_selected_color_space(cv2.imread(filename_reference_image))

    def load_annotated_image(self, filename_annotated_image):
        self.annotated_image = cv2.imread(filename_annotated_image)

    def generate_pixel_mask(self, lower_range=(0, 0, 245), higher_range=(10, 10, 256)):
        ic(self.annotated_image)
        self.pixel_mask = cv2.inRange(self.annotated_image, lower_range, higher_range)
        pixels = np.reshape(self.reference_image, (-1, 3))
        mask_pixels = np.reshape(self.pixel_mask, (-1))
        self.values = pixels[mask_pixels == 255,].transpose()

    def show_statistics_of_pixel_mask(self):
        print(f"Number of annotated pixels: { self.values.shape }")
        if self.values.shape[1] < 100:
            raise Exception("Not enough annotated pixels")

    def save_pixel_values_to_file(self, filename):
        print(f'Writing pixel values to the file "{ filename }"')
        np.savetxt(
            filename,
            self.values.transpose(),
            delimiter="\t",
            fmt="%i",
            header=self.color_space.color_space[0]
            + "\t"
            + self.color_space.color_space[1]
            + "\t"
            + self.color_space.color_space[2],
            comments="",
        )


class ColorBasedSegmenter:
    def __init__(self):
        self.output_tile_location = None
        self.color_space = ColorSpace()
        self.reference_pixels = ReferencePixels(self.color_space)
        self.colormodel = MahalanobisDistance()
        self.ref_image_filename = None
        self.ref_image_annotated_filename = None
        self.ref_image_annotated_is_black_and_white = False
        self.output_scale_factor = None
        self.pixel_mask_file = "pixel_values"
        self.image_statistics = np.zeros(256)

    def main(self, tile_list):
        self.initialize_color_model(self.ref_image_filename, self.ref_image_annotated_filename)
        start = time.time()
        for tile in tqdm(tile_list):
            tile.img = self.color_space.convert_to_selected_color_space(tile.img)
            self.process_tile(tile)
        print("Time to run all tiles: ", time.time() - start)
        """start = time.time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self.process_tile, tile_list)
        print("Time to run all tiles: ", time.time() - start)"""
        self.calculate_statistics(tile_list)
        self.save_statistics()

    def is_image_empty(self, image):
        """Helper function for deciding if an image contains no data."""
        return np.max(image[:, :, 0]) == np.min(image[:, :, 0])

    def ensure_parent_directory_exist(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

    def initialize_color_model(self, ref_image_filename, ref_image_annotated_filename):
        self.reference_pixels.load_reference_image(ref_image_filename)
        self.reference_pixels.load_annotated_image(ref_image_annotated_filename)
        if self.ref_image_annotated_is_black_and_white:
            self.reference_pixels.generate_pixel_mask(lower_range=(245, 245, 245), higher_range=(256, 256, 256))
        else:
            self.reference_pixels.generate_pixel_mask()
        self.reference_pixels.show_statistics_of_pixel_mask()
        self.ensure_parent_directory_exist(self.output_tile_location)
        self.reference_pixels.save_pixel_values_to_file(self.output_tile_location + "/" + self.pixel_mask_file + ".csv")
        self.colormodel.calculate_statistics(self.reference_pixels.values)
        self.colormodel.show_statistics()

    def process_tile(self, tile):
        if not self.is_image_empty(tile.img):
            distance_image = self.colormodel.calculate_distance(tile.img[:, :, :])
            distance = cv2.convertScaleAbs(distance_image, alpha=self.output_scale_factor, beta=0)
            distance = distance.astype(np.uint8)
            tile.img = distance
            tile.save_tile(self.output_tile_location)

    def calculate_statistics(self, tile_list):
        null_dist = self.colormodel.calculate_distance(np.ones((1, 1, 3)) * 255)[0][0]
        for tile in tile_list:
            if np.max(tile.img[:, :]) != np.min(tile.img[:, :]):
                image_statistics = np.histogram(tile.img, bins=256, range=(0, 255))[0]
                # Empty pixel are not counted in the histogram.
                # Unwanted side effect is that pixels with a similar distance will also be discarded.
                image_statistics[int(null_dist * self.output_scale_factor)] = 0
                self.image_statistics += image_statistics
        mean_divide = 0
        mean_sum = 0
        for x in range(0, 256):
            mean_sum += self.image_statistics[x] * x
            mean_divide += self.image_statistics[x]
        ic(null_dist)
        self.mean_pixel_value = mean_sum / mean_divide

    def save_statistics(self):
        statistics_path = self.output_tile_location + "/statistics"
        self.ensure_parent_directory_exist(statistics_path)
        print(f'Writing statistics to the folder "{ statistics_path }"')
        # Plot histogram of pixel values
        plt.plot(self.image_statistics)
        plt.title("Histogram of pixel values")
        plt.xlabel("Pixel Value")
        plt.ylabel("Number of Pixels")
        plt.savefig(statistics_path + "/Histogram of pixel values", dpi=300)
        plt.close()
        with open(statistics_path + "/output_file.txt", "w") as f:
            f.write("Input parameters:\n")
            f.write(f" - Orthomosaic: {args.orthomosaic}\n")
            f.write(f" - Reference image: {args.reference}\n")
            f.write(f" - Annotated image: {args.annotated}\n")
            f.write(f" - Output scale factor: {args.scale}\n")
            f.write(f" - Tile sizes: {args.tile_size}\n")
            f.write(f" - Output tile location: {args.output_tile_location}\n")
            f.write(f" - Method: {args.method}\n")
            f.write(f" - Parameter: {args.param}\n")
            f.write(f" - Colorspace: {args.colorspace}\n")
            f.write(f" - Pixel mask file: {args.mask_file_name}\n")
            f.write(f" - Date and time of execution: {datetime.now().replace(microsecond=0)}\n")
            f.write("\n\nOutput from run\n")
            f.write(" - Average color value of annotated pixels\n")
            f.write(f" - {self.colormodel.average}\n")
            f.write(" - Covariance matrix of the annotated pixels\n")
            f.write(" - " + str(self.colormodel.covariance).replace("\n", "\n   ") + "\n")
            f.write(f" - Mean pixel value: {self.mean_pixel_value}\n")
            f.write(f" - Number of tiles: {len(tile_list)}\n")


parser = argparse.ArgumentParser(
    prog="ColorDistranceCalculatorForOrthomosaics",
    description="A tool for calculating color distances in an "
    "orthomosaic to a reference color based on samples from "
    "an annotated image.",
    epilog="Program written by Henrik Skov Midtiby (hemi@mmmi.sdu.dk) in "
    "2023 as part of the Precisionseedbreeding project supported "
    "by GUDP and Frøafgiftsfonden.",
)
parser.add_argument("orthomosaic", help="Path to the orthomosaic that you want to process.")
parser.add_argument("reference", help="Path to the reference image.")
parser.add_argument("annotated", help="Path to the annotated reference image.")
parser.add_argument(
    "--scale",
    default=5,
    type=float,
    help="The calculated distances are multiplied with this "
    "factor before the result is saved as an image. "
    "Default value is 5.",
)
parser.add_argument(
    "--tile_size", default=3000, type=int, help="The height and width of tiles that are analyzed. " "Default is 3000."
)
parser.add_argument(
    "--output_tile_location", default="output/mahal", help="The location in which to save the mahalanobis tiles."
)
parser.add_argument("--input_tile_location", default=None, help="The location in which to save the input tiles.")
parser.add_argument(
    "--method",
    default="mahalanobis",
    help="The method used for calculating distances from the "
    "set of annotated pixels. "
    "Possible values are 'mahalanobis' for using the "
    "Mahalanobis distance and "
    "'gmm' for using a Gaussian Mixture Model."
    "'mahalanobis' is the default value.",
)
parser.add_argument(
    "--param",
    default=2,
    type=int,
    help="Numerical parameter for the color model. "
    "When using the 'gmm' method, this equals the "
    "number of components in the Gaussian Mixture Model.",
)
parser.add_argument(
    "--colorspace",
    default="bgr",
    help="Defines which colorspace will be used to find specific "
    "colors in an orthomosaic. \n"
    "Default is bgr(rgb), but cielab can be chosen with lab "
    "and HSV can be chosen with hsv",
)
"""parser.add_argument('--process_tiles',
                    default=0,
                    help='Defaults to 0 which means that the tiles are not '
                         'processed, but a .csv file with the mask pixels is '
                         'returned')"""
parser.add_argument(
    "--mask_file_name",
    default="pixel_values",
    help="Change the name in which the pixel mask is saved. It "
    "defaults to pixel_values (.csv is automatically added)",
)
parser.add_argument(
    "--annotated_bw",
    action="store_true",
    help="Enable if the annotated reference image is black with white annotations instead of the default image with red annotations.",
)
parser.add_argument(
    "--run_specific_tile",
    nargs="+",
    type=int,
    metavar="TILE_ID",
    help="If set, only run the specific tile numbers. (--run_specific_tile 16 65) will run tile 16 and 65.",
)
parser.add_argument(
    "--run_specific_tileset",
    nargs="+",
    type=int,
    metavar="FROM_TILE_ID TO_TILE_ID",
    help="takes two inputs like (--from_specific_tileset 16 65). This will run every tile from 16 to 65.",
)
args = parser.parse_args()


# Initialize the tile separator
tsr = convert_orthomosaic_to_list_of_tiles()
tsr.run_specific_tile = args.run_specific_tile
tsr.run_specific_tileset = args.run_specific_tileset
tsr.tile_size = args.tile_size
tsr.output_tile_location = args.output_tile_location
tile_list = tsr.main(args.orthomosaic)

cbs = ColorBasedSegmenter()
if args.method == "gmm":
    cbs.colormodel = GaussianMixtureModelDistance(args.param)
cbs.output_tile_location = args.output_tile_location
cbs.ref_image_filename = args.reference
cbs.ref_image_annotated_filename = args.annotated
cbs.output_scale_factor = args.scale
cbs.pixel_mask_file = args.mask_file_name
cbs.color_space = ColorSpace(args.colorspace)
cbs.ref_image_annotated_is_black_and_white = args.annotated_bw
cbs.main(tile_list)


# python3 color_based_segmenter.py Tests/rødsvingel/input_data/2023-04-03_Rødsvingel_1._års_Wagner_JSJ_2_ORTHO.tif Tests/rødsvingel/input_data/original.png Tests/rødsvingel/input_data/annotated.png --output_tile_location Tests/rødsvingel/tiles --tile_size 500
