"""Make a color model based on reference pixel color values."""

from __future__ import annotations

import os
import pathlib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import rasterio
from numpy.typing import NDArray
from sklearn import mixture

from OCDC.transforms import BaseTransform


class ReferencePixels:
    """
    A Class for handling the reference pixels for color models.

    Parameters
    ----------
    reference
        Reference image from which the pixels are extracted.
    annotated
        Image with annotated pixels locations for extraction.
    bands_to_use
        Which bands to extract pixel values from.
    alpha_channel
        Which channel is the alpha channel.
    transform
        A transform to apply on the extracted pixels.
    **kwargs
        Not used.
    """

    def __init__(
        self,
        *,
        reference: pathlib.Path,
        annotated: pathlib.Path,
        bands_to_use: tuple[int, ...] | list[int] | None = None,
        alpha_channel: int | None = -1,
        transform: BaseTransform | None = None,
        **kwargs: Any,
    ):
        self.reference_image_filename = reference
        self.mask_filename = annotated
        self.bands_to_use: tuple[int, ...] | list[int] | None = bands_to_use
        self.alpha_channel: int | None = alpha_channel
        self.transform: BaseTransform | None = transform
        self.reference_image: NDArray[Any] = np.zeros(0)
        self.mask: NDArray[Any] = np.zeros(0)
        self.values: NDArray[Any] = np.zeros(0)
        """The reference pixel values."""
        self._initialize()

    def _initialize(self) -> None:
        self._load_reference_image(self.reference_image_filename)
        self._load_mask(self.mask_filename)
        self._get_bands_to_use()
        self._generate_pixel_mask()
        self._show_statistics_of_pixel_mask()

    def _get_bands_to_use(self) -> None:
        if self.bands_to_use is None:
            self.bands_to_use = tuple(range(self.reference_image.shape[0]))
            if self.alpha_channel is not None:
                if self.alpha_channel < -1 or self.alpha_channel > self.reference_image.shape[0] - 1:
                    raise ValueError(
                        f"Alpha channel have to be between -1 and {self.reference_image.shape[0]-1}, but got {self.alpha_channel}."
                    )
                elif self.alpha_channel == -1:
                    self.alpha_channel = self.reference_image.shape[0] - 1
                self.bands_to_use = tuple(x for x in self.bands_to_use if x != self.alpha_channel)
        for band in self.bands_to_use:
            if band < 0 or band > self.reference_image.shape[0] - 1:
                raise ValueError(f"Bands have to be between 0 and {self.reference_image.shape[0]-1}, but got {band}.")

    def _load_reference_image(self, filename_reference_image: pathlib.Path) -> None:
        with rasterio.open(filename_reference_image) as ref_img:
            self.reference_image = ref_img.read()
        if self.transform is not None:
            self.reference_image = self.transform.transform(self.reference_image)

    def _load_mask(self, filename_mask: pathlib.Path) -> None:
        with rasterio.open(filename_mask) as msk:
            self.mask = msk.read()

    def _generate_pixel_mask(
        self, lower_range: tuple[int, int, int] = (245, 0, 0), higher_range: tuple[int, int, int] = (256, 10, 10)
    ) -> None:
        if self.mask.shape[0] == 3 or self.mask.shape[0] == 4:
            pixel_mask = np.where(
                (self.mask[0, :, :] >= lower_range[0])
                & (self.mask[0, :, :] <= higher_range[0])
                & (self.mask[1, :, :] >= lower_range[1])
                & (self.mask[1, :, :] <= higher_range[1])
                & (self.mask[2, :, :] >= lower_range[2])
                & (self.mask[2, :, :] <= higher_range[2]),
                255,
                0,
            )
        elif self.mask.shape[0] == 1:
            pixel_mask = np.where((self.mask[0, :, :] > 127), 255, 0)
        else:
            raise TypeError(f"Expected a Black and White or RGB(A) image for mask but got {self.mask.shape[0]} Bands")
        self.values = self.reference_image[:, pixel_mask == 255]
        self.values = self.values[self.bands_to_use, :]

    def _show_statistics_of_pixel_mask(self) -> None:
        print(f"Number of annotated pixels: { self.values.shape }")
        min_annotated_pixels = 100
        if self.values.shape[1] <= min_annotated_pixels:
            raise Exception(
                f"Not enough annotated pixels. Need at least {min_annotated_pixels}, but got {self.values.shape[1]}"
            )

    def save_pixel_values_to_file(self, filename: pathlib.Path) -> None:
        """Save pixel values to csv file with tab delimiter."""
        output_directory = os.path.dirname(filename)
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
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


class BaseDistance(ABC):
    """
    Base class for all color distance models.

    Parameters
    ----------
    **kwargs
        Not used.
    """

    def __init__(self, **kwargs: Any):
        self.reference_pixels = ReferencePixels(**kwargs)
        self.bands_to_use = self.reference_pixels.bands_to_use
        self.covariance: NDArray[Any]
        """Covariance of the reference pixels."""
        self.average: float
        """Average of the reference pixels."""
        self._initialize()

    def _initialize(self) -> None:
        self._calculate_statistics()
        self.show_statistics()

    def save_pixel_values(self, filename: pathlib.Path) -> None:
        """Save reference pixels to a csv file."""
        self.reference_pixels.save_pixel_values_to_file(filename)

    @abstractmethod
    def _calculate_statistics(self) -> None:
        pass

    @abstractmethod
    def calculate_distance(self, image: NDArray[Any]) -> NDArray[Any]:
        """Calculate the color distance for each pixel in the image to the reference."""
        if self.reference_pixels.transform is not None:
            image = self.reference_pixels.transform.transform(image)
        return image

    @abstractmethod
    def show_statistics(self) -> None:
        """Print the statistics to screen."""
        pass


class MahalanobisDistance(BaseDistance):
    """
    A multivariate normal distribution used to describe the color of a set of
    pixels.

    Parameters
    ----------
    **kwargs
        Not used.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def _calculate_statistics(self) -> None:
        self.covariance: NDArray[Any] = np.cov(self.reference_pixels.values)
        self.average = np.average(self.reference_pixels.values, axis=1)

    def calculate_distance(self, image: NDArray[Any]) -> NDArray[Any]:
        """
        Calculate the color distance using mahalanobis
        for each pixel in the image to the reference.
        """
        image = super().calculate_distance(image)
        assert self.bands_to_use is not None
        pixels = np.reshape(image[self.bands_to_use, :, :], (len(self.bands_to_use), -1)).transpose()
        inv_cov = np.linalg.inv(self.covariance)
        diff = pixels - self.average
        modified_dot_product = diff * (diff @ inv_cov)
        distance = np.sum(modified_dot_product, axis=1)
        distance = np.sqrt(distance)
        distance_image = np.reshape(distance, (1, image.shape[1], image.shape[2]))
        return distance_image

    def show_statistics(self) -> None:
        print("Average color value of annotated pixels")
        print(self.average)
        print("Covariance matrix of the annotated pixels")
        print(self.covariance)


class GaussianMixtureModelDistance(BaseDistance):
    """
    A Gaussian Mixture Model from sklearn where the loglikelihood is converted
    to a distance with so output is similar til mahalanobis.

    Parameters
    ----------
    n_components
        The number of mixture components.
    **kwargs
        Not used.
    """

    def __init__(self, n_components: int, **kwargs: Any):
        self.n_components = n_components
        super().__init__(**kwargs)

    def _calculate_statistics(self) -> None:
        self.gmm = mixture.GaussianMixture(n_components=self.n_components, covariance_type="full")
        self.gmm.fit(self.reference_pixels.values.transpose())
        self.average = self.gmm.means_
        self.covariance = self.gmm.covariances_
        self.min_score = np.min(self.gmm.score_samples(self.average))

    def calculate_distance(self, image: NDArray[Any]) -> NDArray[Any]:
        """
        Calculate the color distance using a gaussian mixture model
        for each pixel in the image to the reference.
        """
        image = super().calculate_distance(image)
        assert self.bands_to_use is not None
        pixels = np.reshape(image[self.bands_to_use, :, :], (len(self.bands_to_use), -1)).transpose()
        loglikelihood = self.gmm.score_samples(pixels)
        distance = np.sqrt(np.maximum(-(loglikelihood - self.min_score), 0))
        distance_image = np.reshape(distance, (1, image.shape[1], image.shape[2]))
        return distance_image

    def show_statistics(self) -> None:
        print("GMM")
        print(self.gmm)
        print(self.average)
        print(self.covariance)
