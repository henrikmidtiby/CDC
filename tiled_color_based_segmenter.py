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

# fix make it print less and only what makes sense.
# all other can be saved as output or maybe add an arg to verbose.

# bw finds more pixels so maybe this should be addressed.
# BW: 3552 RGB: 316

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from orthomosaic_tiler import OrthomosaicTiles


class TiledColorBasedSegmenter:
    def __init__(self, *, color_model, bands_to_use, scale, output_tile_location, **kwargs):
        self.ortho_tiler = OrthomosaicTiles(**kwargs)
        self.bands_to_use = bands_to_use
        self.output_tile_location = output_tile_location
        self.colormodel = color_model
        self.output_scale_factor = scale
        self.image_statistics = np.zeros(256)
        self.ortho_tiler.divide_orthomosaic_into_tiles()

    def convertScaleAbs(self, img, alpha):
        scaled_img = np.minimum(np.abs(alpha * img), 255)
        return scaled_img

    def is_image_empty(self, image):
        """Helper function for deciding if an image contains no data."""
        return np.max(image[:, :, 0]) == np.min(image[:, :, 0])

    def process_image(self, image):
        if not self.is_image_empty(image):
            distance_image = self.colormodel.calculate_distance(image)
            distance = self.convertScaleAbs(distance_image, alpha=self.output_scale_factor)
            distance = distance.astype(np.uint8)
            return distance

    def process_tiles(self, save_tiles=True, save_ortho=True):
        for tile in tqdm(self.ortho_tiler.tiles):
            img, _ = tile.read_tile(self.ortho_tiler.orthomosaic, self.bands_to_use)
            distance_img = self.process_image(img)
            if save_tiles:
                tile.save_tile(distance_img, self.output_tile_location)
            tile.output = distance_img
        if save_ortho:
            output_filename = self.output_tile_location.joinpath("orthomosaic.tiff")
            self.ortho_tiler.save_orthomosaic_from_tile_output(output_filename)

    def calculate_statistics(self):
        for tile in self.ortho_tiler.tiles:
            output = np.where(tile.mask > 0, tile.output, np.nan)
            if np.max(output) != np.min(output):
                image_statistics = np.histogram(output, bins=256, range=(0, 255))[0]
                self.image_statistics += image_statistics
        mean_divide = 0
        mean_sum = 0
        for x in range(0, 256):
            mean_sum += self.image_statistics[x] * x
            mean_divide += self.image_statistics[x]
        self.mean_pixel_value = mean_sum / mean_divide

    def save_statistics(self, args):
        statistics_path = self.output_tile_location.joinpath("statistics")
        print(f'Writing statistics to the folder "{ statistics_path }"')
        # Plot histogram of pixel values
        plt.plot(self.image_statistics)
        plt.title("Histogram of pixel values")
        plt.xlabel("Pixel Value")
        plt.ylabel("Number of Pixels")
        histrogram_filename = statistics_path.joinpath("Histogram of pixel values")
        output_directory = os.path.dirname(histrogram_filename)
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
        plt.savefig(histrogram_filename, dpi=300)
        plt.close()
        with open(statistics_path.joinpath("output_file.txt"), "w") as f:
            f.write("Input parameters:\n")
            f.write(f" - Orthomosaic: {args.orthomosaic}\n")
            f.write(f" - Reference image: {args.reference}\n")
            f.write(f" - Annotated image: {args.annotated}\n")
            f.write(f" - Output scale factor: {args.scale}\n")
            f.write(f" - Tile sizes: {args.tile_size}\n")
            f.write(f" - Output tile location: {args.output_tile_location}\n")
            f.write(f" - Method: {args.method}\n")
            f.write(f" - Parameter: {args.param}\n")
            f.write(f" - Pixel mask file: {args.mask_file_name}\n")
            f.write(f" - Date and time of execution: {datetime.now().replace(microsecond=0)}\n")
            f.write("\n\nOutput from run\n")
            f.write(" - Average color value of annotated pixels\n")
            f.write(f" - {self.colormodel.average}\n")
            f.write(" - Covariance matrix of the annotated pixels\n")
            f.write(" - " + str(self.colormodel.covariance).replace("\n", "\n   ") + "\n")
            f.write(f" - Mean pixel value: {self.mean_pixel_value}\n")
            # f.write(f" - Number of tiles: {len(tile_list)}\n")
