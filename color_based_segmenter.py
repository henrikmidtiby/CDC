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

import os
import time

# import matplotlib.pyplot as plt
import numpy as np
import rasterio
from tqdm import tqdm

from color_models import MahalanobisDistance


def convertScaleAbs(img, alpha):
    scaled_img = np.minimum(np.abs(alpha * img), 255)
    return scaled_img


class ReferencePixels:
    def __init__(self):
        self.reference_image = None
        self.mask = None
        self.values = None
        self.bands_to_use = None

    def load_reference_image(self, filename_reference_image):
        with rasterio.open(filename_reference_image) as ref_img:
            self.reference_image = ref_img.read()

    def load_mask(self, filename_mask):
        with rasterio.open(filename_mask) as msk:
            self.mask = msk.read()

    def generate_pixel_mask(self, lower_range=(245, 0, 0), higher_range=(256, 10, 10)):
        if self.mask.shape[0] == 3:
            pixel_mask = np.where(
                (self.mask[0, :, :] > lower_range[0])
                & (self.mask[0, :, :] < higher_range[0])
                & (self.mask[1, :, :] > lower_range[1])
                & (self.mask[1, :, :] < higher_range[1])
                & (self.mask[2, :, :] > lower_range[2])
                & (self.mask[2, :, :] < higher_range[2]),
                255,
                0,
            )
        elif self.mask.shape[0] == 1:
            pixel_mask = np.where((self.mask[0, :, :] > 128), 255, 0)
        else:
            raise Exception(f"Expected a Black and White or RGB image for mask but got {self.mask.shape[0]} Bands")
        self.values = self.reference_image[:, pixel_mask == 255]
        self.values = self.values[self.bands_to_use, :]

    def show_statistics_of_pixel_mask(self):
        print(f"Number of annotated pixels: { self.values.shape }")
        min_annotated_pixels = 100
        if self.values.shape[1] <= min_annotated_pixels:
            raise Exception(
                f"Not enough annotated pixels. Need at least {min_annotated_pixels}, but got {self.values.shape[1]}"
            )

    def save_pixel_values_to_file(self, filename):
        print(f'Writing pixel values to the file "{ filename }"')
        np.savetxt(
            filename,
            self.values.transpose(),
            delimiter="\t",
            # fmt="%i",
            # header=self.color_space.color_space[0]
            # + "\t"
            # + self.color_space.color_space[1]
            # + "\t"
            # + self.color_space.color_space[2],
            comments="",
        )


class ColorBasedSegmenter:
    def __init__(self):
        self.output_tile_location = None
        self.reference_pixels = ReferencePixels()
        self.colormodel = MahalanobisDistance()
        self.bands_to_use = None
        self.ref_image_filename = None
        self.ref_image_annotated_filename = None
        self.output_scale_factor = None
        self.pixel_mask_file = "pixel_values"
        self.image_statistics = np.zeros(256)

    def main(self, tile_list):
        self.initialize_color_model(self.ref_image_filename, self.ref_image_annotated_filename)
        start = time.time()
        for tile in tqdm(tile_list):
            self.process_tile(tile)
        print("Time to run all tiles: ", time.time() - start)
        """start = time.time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self.process_tile, tile_list)
        print("Time to run all tiles: ", time.time() - start)"""
        # self.calculate_statistics(tile_list)
        # self.save_statistics()

    def is_image_empty(self, image):
        """Helper function for deciding if an image contains no data."""
        return np.max(image[:, :, 0]) == np.min(image[:, :, 0])

    def ensure_parent_directory_exist(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

    def initialize_color_model(self):
        self.reference_pixels.load_reference_image(self.ref_image_filename)
        self.reference_pixels.load_mask(self.ref_image_annotated_filename)
        if self.bands_to_use is None:
            self.bands_to_use = tuple(range(self.reference_pixels.reference_image.shape[0] - 1))
        self.reference_pixels.bands_to_use = self.bands_to_use
        self.reference_pixels.generate_pixel_mask()
        self.reference_pixels.show_statistics_of_pixel_mask()
        self.ensure_parent_directory_exist(self.output_tile_location)
        self.reference_pixels.save_pixel_values_to_file(self.output_tile_location + "/" + self.pixel_mask_file + ".csv")
        self.colormodel.bands_to_use = self.bands_to_use
        self.colormodel.calculate_statistics(self.reference_pixels.values)
        self.colormodel.show_statistics()

    def process_image(self, image):
        if not self.is_image_empty(image):
            distance_image = self.colormodel.calculate_distance(image)
            distance = convertScaleAbs(distance_image, alpha=self.output_scale_factor)
            distance = distance.astype(np.uint8)
            return distance

    def process_tile(self, tile):
        if not self.is_image_empty(tile.img):
            distance_image = self.colormodel.calculate_distance(tile.img)
            distance = convertScaleAbs(distance_image, alpha=self.output_scale_factor)
            distance = distance.astype(np.uint8)
            tile.img = distance
            tile.save_tile(self.output_tile_location)

    def calculate_statistics(self, tile_list):
        null_dist = self.colormodel.calculate_distance(np.ones((len(self.bands_to_use), 1, 1)) * 255)[0][0]
        for tile in tile_list:
            if np.max(tile.img) != np.min(tile.img):
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
        self.mean_pixel_value = mean_sum / mean_divide

    # def save_statistics(self):
    #     statistics_path = self.output_tile_location + "/statistics"
    #     self.ensure_parent_directory_exist(statistics_path)
    #     print(f'Writing statistics to the folder "{ statistics_path }"')
    #     # Plot histogram of pixel values
    #     plt.plot(self.image_statistics)
    #     plt.title("Histogram of pixel values")
    #     plt.xlabel("Pixel Value")
    #     plt.ylabel("Number of Pixels")
    #     plt.savefig(statistics_path + "/Histogram of pixel values", dpi=300)
    #     plt.close()
    #     with open(statistics_path + "/output_file.txt", "w") as f:
    #         f.write("Input parameters:\n")
    #         f.write(f" - Orthomosaic: {args.orthomosaic}\n")
    #         f.write(f" - Reference image: {args.reference}\n")
    #         f.write(f" - Annotated image: {args.annotated}\n")
    #         f.write(f" - Output scale factor: {args.scale}\n")
    #         f.write(f" - Tile sizes: {args.tile_size}\n")
    #         f.write(f" - Output tile location: {args.output_tile_location}\n")
    #         f.write(f" - Method: {args.method}\n")
    #         f.write(f" - Parameter: {args.param}\n")
    #         f.write(f" - Colorspace: {args.colorspace}\n")
    #         f.write(f" - Pixel mask file: {args.mask_file_name}\n")
    #         f.write(f" - Date and time of execution: {datetime.now().replace(microsecond=0)}\n")
    #         f.write("\n\nOutput from run\n")
    #         f.write(" - Average color value of annotated pixels\n")
    #         f.write(f" - {self.colormodel.average}\n")
    #         f.write(" - Covariance matrix of the annotated pixels\n")
    #         f.write(" - " + str(self.colormodel.covariance).replace("\n", "\n   ") + "\n")
    #         f.write(f" - Mean pixel value: {self.mean_pixel_value}\n")
    #         f.write(f" - Number of tiles: {len(tile_list)}\n")
