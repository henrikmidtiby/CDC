"""Colorbased distance calculation on tiles."""

from __future__ import annotations

import os
import pathlib
import threading
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.enums import Resampling
from tqdm.contrib.concurrent import thread_map

from CDC.color_models import BaseDistance
from CDC.orthomosaic_tiler import OrthomosaicTiles, Tile


class TiledColorBasedDistance:
    """
    Calculate color based distance on tiled orthomosaic.

    Parameters
    ----------
    ortho_tiler
        An instance of :class:`~CDC.orthomosaic_tiler.OrthomosaicTiles`
    color_model
        The color model to use for distance calculations. See :mod:`~CDC.color_models`
    scale
        A scale factor to scale the calculated distances with.
    output_location
        Where output orthomosaic and tiles are saved.
    """

    def __init__(
        self,
        *,
        ortho_tiler: OrthomosaicTiles,
        color_model: BaseDistance,
        scale: float,
        output_location: pathlib.Path,
    ):
        self.ortho_tiler = ortho_tiler
        self.output_location = output_location
        self.colormodel = color_model
        self.output_scale_factor = scale
        self.ortho_tiler.divide_orthomosaic_into_tiles()

    @staticmethod
    def convertScaleAbs(image: NDArray[Any], alpha: float) -> NDArray[Any]:
        """Scale image by alpha and take the absolute value."""
        scaled_img: NDArray[Any] = np.minimum(np.abs(alpha * image), 255)
        return scaled_img

    def process_image(self, image: NDArray[Any]) -> NDArray[Any]:
        """Calculate the color based distance on image."""
        distance_image = self.colormodel.calculate_distance(image)
        distance = self.convertScaleAbs(distance_image, alpha=self.output_scale_factor)
        distance = distance.astype(np.uint8)
        return distance

    def process_tiles(self, save_tiles: bool = False, max_workers: int | None = os.cpu_count()) -> None:
        """
        Calculate color based distance on all tiles and save output.

        Parameters
        ----------
        save_tiles
            Save all tiles to output_location.
        max_workers
            Maximum number of threads to use for processing.
        """
        if max_workers is None:
            max_workers = 1
        read_lock = threading.Lock()
        write_lock = threading.Lock()
        output_filename = self.output_location.joinpath("orthomosaic.tiff")
        with rasterio.open(self.ortho_tiler.orthomosaic) as src:
            profile = src.profile
            profile["count"] = 1
            overview_factors = src.overviews(src.indexes[0])
            with rasterio.open(output_filename, "w", **profile) as dst:

                def process(tile: Tile) -> None:
                    with read_lock:
                        img = src.read(window=tile.window_with_overlap)
                        mask_temp = src.read_masks(window=tile.window_with_overlap)
                    mask = mask_temp[0]
                    for band in range(mask_temp.shape[0]):
                        mask = mask & mask_temp[band]
                    distance_img = self.process_image(img)
                    with write_lock:
                        output = tile.get_window_pixels(distance_img)
                        dst.write(output, window=tile.window)
                        mask = tile.get_window_pixels(np.expand_dims(mask, 0)).squeeze()
                        dst.write_mask(mask, window=tile.window)
                    if save_tiles:
                        tile.save_tile(distance_img, mask, self.output_location)

                thread_map(process, self.ortho_tiler.tiles, max_workers=max_workers)
        with rasterio.open(output_filename, "r+") as dst:
            dst.build_overviews(overview_factors, Resampling.average)

    def _calculate_statistics(self) -> tuple[NDArray[Any], float]:
        image_statistics = np.zeros(256)
        for tile in self.ortho_tiler.tiles:
            output = np.where(tile.mask > 0, tile.output, np.nan)
            if np.max(output) != np.min(output):
                image_statistics += np.histogram(output, bins=256, range=(0, 255))[0]
        mean_divide = 0
        mean_sum = 0
        for x in range(0, 256):
            mean_sum += image_statistics[x] * x
            mean_divide += image_statistics[x]
        mean_pixel_value = mean_sum / mean_divide
        return image_statistics, mean_pixel_value

    def save_statistics(self, args: Any) -> None:
        """
        Calculate a histogram of the color based distance from all tiles.
        Save histogram in output_location/statistics with a txt file of metadata.
        """
        histogram, mean_pixel_value = self._calculate_statistics()
        statistics_path = self.output_location.joinpath("statistics")
        print(f'Writing statistics to the folder "{statistics_path}"')
        # Plot histogram of pixel values
        plt.plot(histogram)
        plt.title("Histogram of pixel values")
        plt.xlabel("Pixel Value")
        plt.ylabel("Number of Pixels")
        histogram_filename = statistics_path.joinpath("Histogram of pixel values")
        output_directory = os.path.dirname(histogram_filename)
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
        plt.savefig(histogram_filename, dpi=300)
        plt.close()
        with open(statistics_path.joinpath("statistics.txt"), "w") as f:
            f.write("Input parameters:\n")
            f.write(f" - Orthomosaic: {args.orthomosaic}\n")
            f.write(f" - Reference image: {args.reference}\n")
            f.write(f" - Annotated image: {args.annotated}\n")
            f.write(f" - Output scale factor: {args.scale}\n")
            f.write(f" - Tile sizes: {args.tile_size}\n")
            f.write(f" - Output tile location: {args.output_location}\n")
            f.write(f" - Method: {args.method}\n")
            f.write(f" - Parameter: {args.param}\n")
            f.write(f" - Date and time of execution: {datetime.now().replace(microsecond=0)}\n")
            f.write("\n\nOutput from run\n")
            f.write(" - Average color value of annotated pixels\n")
            f.write(f" - {self.colormodel.average}\n")
            f.write(" - Covariance matrix of the annotated pixels\n")
            f.write(" - " + str(self.colormodel.covariance).replace("\n", "\n   ") + "\n")
            f.write(f" - Mean pixel value: {mean_pixel_value}\n")
            f.write(f" - Number of tiles: {len(self.ortho_tiler.tiles)}\n")
