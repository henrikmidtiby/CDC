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

from CDC.transforms import BaseTransform


class ReferencePixels:
    """
    A Class for handling the reference pixels for color models.
    Extracted from a reference image and an annotated mask.

    Parameters
    ----------
    reference
        Reference image from which the pixels are extracted.
    annotated
        Image with annotated pixels locations for extraction.
    """

    def __init__(self, *, reference: pathlib.Path, annotated: pathlib.Path):
        self.values: NDArray[Any] = np.zeros(0)
        """The reference pixel values."""
        ref_image = self.load_image(reference)
        mask_image = self.load_image(annotated)
        self.generate_pixel_values(ref_image, mask_image)

    @staticmethod
    def load_image(file_name: pathlib.Path) -> NDArray[Any]:
        """Load image from file."""
        try:
            with rasterio.open(file_name) as img:
                return img.read()
        except Exception as e:
            raise OSError(f"Could not open the image at '{ file_name }'") from e

    def generate_pixel_values(
        self,
        ref_image: NDArray[Any],
        mask_image: NDArray[Any],
        lower_range: tuple[int, int, int] = (245, 0, 0),
        higher_range: tuple[int, int, int] = (256, 10, 10),
    ) -> None:
        """
        Generate Pixel values from reference and mask image.
        Lower_range and higher_range is only used if mask is annotated with a color
        to extract the pixel with a given color. Default red.
        """
        if mask_image.shape[0] == 3 or mask_image.shape[0] == 4:
            pixel_mask = np.where(
                (mask_image[0, :, :] >= lower_range[0])
                & (mask_image[0, :, :] <= higher_range[0])
                & (mask_image[1, :, :] >= lower_range[1])
                & (mask_image[1, :, :] <= higher_range[1])
                & (mask_image[2, :, :] >= lower_range[2])
                & (mask_image[2, :, :] <= higher_range[2]),
                255,
                0,
            )
        elif mask_image.shape[0] == 1:
            pixel_mask = np.where((mask_image[0, :, :] > 127), 255, 0)
        else:
            raise TypeError(f"Expected a Black and White or RGB(A) image for mask but got {mask_image.shape[0]} Bands")
        self.values = ref_image[:, pixel_mask == 255]
        min_annotated_pixels = 100
        if self.values.shape[1] <= min_annotated_pixels:
            raise Exception(
                f"Not enough annotated pixels. Need at least {min_annotated_pixels}, but got {self.values.shape[1]}"
            )


class BaseDistance(ABC):
    """
    Base class for all color distance models.
    Can be used to create new methods for calculating the color distance.
    Sub classes must implement :meth:`~CDC.color_models.calculate_distance`,
    :meth:`~CDC.color_models.calculate_statistics` and
    :meth:`~CDC.color_models.show_statistics`.

    Parameters
    ----------
    reference_pixels
        Pixels to use as a reference
    bands_to_use
        A list of indexes to choose which "colors" are used in distance calculations
    transform
        A transform to apply to the images before the distance is calculated.
    """

    def __init__(
        self,
        *,
        reference_pixels: NDArray[Any],
        bands_to_use: tuple[int, ...] | list[int] | None = None,
        transform: BaseTransform | None = None,
    ):
        self.color_values: NDArray[Any] = reference_pixels
        """Reference pixel values"""
        self.bands_to_use: tuple[int, ...] | list[int] | None = bands_to_use
        self.transform: BaseTransform | None = transform
        self.color_values_raw: NDArray[Any] | None = None
        """Raw pixel values as reference"""
        self.color_values_transformed: NDArray[Any] | None = None
        """Transformed reference pixel values"""
        self.covariance: NDArray[Any] | None = None
        """Covariance of the reference pixels."""
        self.average: float | None = None
        """Average of the reference pixels."""
        self.calculate_statistics()

    @classmethod
    def from_image_annotation(
        cls,
        *,
        reference: pathlib.Path,
        annotated: pathlib.Path,
        bands_to_use: tuple[int, ...] | list[int] | None = None,
        alpha_channel: int | None = -1,
        transform: BaseTransform | None = None,
        **kwargs: Any,
    ) -> BaseDistance:
        """Create a class instance from a reference image and an annotated mask."""
        ref_pix = ReferencePixels(reference=reference, annotated=annotated)
        cls_instance = cls(reference_pixels=ref_pix.values, bands_to_use=bands_to_use, transform=transform, **kwargs)
        cls_instance.get_bands_to_use(alpha_channel, ref_pix.values.shape[0])
        cls_instance.color_values_raw = ref_pix.values
        if cls_instance.transform is not None:
            cls_instance.color_values_transformed = cls_instance.transform.transform(ref_pix.values)
        else:
            cls_instance.color_values_transformed = ref_pix.values
        cls_instance.color_values = cls_instance.color_values_transformed[cls_instance.bands_to_use, :]
        cls_instance.calculate_statistics()
        return cls_instance

    @classmethod
    def from_pixel_values(
        cls,
        *,
        pixel_values: NDArray[Any],
        bands_to_use: tuple[int, ...] | list[int] | None = None,
        alpha_channel: int | None = -1,
        transform: BaseTransform | None = None,
        **kwargs: Any,
    ) -> BaseDistance:
        """Create a class instance from a list of pixel values before bands_to_use and transforms are applied."""
        cls_instance = cls(reference_pixels=pixel_values, bands_to_use=bands_to_use, transform=transform, **kwargs)
        cls_instance.get_bands_to_use(alpha_channel, pixel_values.shape[0])
        cls_instance.color_values_raw = pixel_values
        if cls_instance.transform is not None:
            cls_instance.color_values_transformed = cls_instance.transform.transform(pixel_values)
        else:
            cls_instance.color_values_transformed = pixel_values
        cls_instance.color_values = cls_instance.color_values_transformed[cls_instance.bands_to_use, :]
        cls_instance.calculate_statistics()
        return cls_instance

    def get_bands_to_use(self, alpha_channel: int | None, number_of_bands: int) -> None:
        """Get correct bands to use from supplied alpha channel number and the number of bands in input."""
        if self.bands_to_use is None:
            self.bands_to_use = tuple(range(number_of_bands))
            if alpha_channel is not None:
                if alpha_channel < -1 or alpha_channel > number_of_bands - 1:
                    raise ValueError(
                        f"Alpha channel have to be between -1 and {number_of_bands-1}, but got {alpha_channel}."
                    )
                elif alpha_channel == -1:
                    alpha_channel = number_of_bands - 1
                self.bands_to_use = tuple(x for x in self.bands_to_use if x != alpha_channel)
        for band in self.bands_to_use:
            if band < 0 or band > number_of_bands - 1:
                raise ValueError(f"Bands have to be between 0 and {number_of_bands-1}, but got {band}.")

    @staticmethod
    def _is_int(array: NDArray[Any]) -> bool:
        diff = np.abs(array - array.astype(int))
        return bool(np.all(diff < 1e-10))

    def save_pixel_values_to_file(
        self, filename: pathlib.Path, values: NDArray[Any], header: str | None = None, raw: bool = True
    ) -> None:
        """Save pixel values to csv file with tab delimiter."""
        output_directory = os.path.dirname(filename)
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
        if self._is_int(values):
            fmt = "%i"
        else:
            fmt = "%f"
        if header is None:
            csv_header = "".join([f"c{x}\t" for x in range(1, values.shape[0] + 1)])[:-1]
        elif raw:
            csv_header = header.replace(",", "\t")
        else:
            header_list = header.split(",")
            csv_header = "".join([f"{header_list[i]}\t" for i in self.bands_to_use])[:-1]  # type: ignore[union-attr]
        print(f'Writing pixel values to the file "{ filename }"')
        np.savetxt(
            filename,
            values.transpose(),
            delimiter="\t",
            fmt=fmt,
            header=csv_header,
            comments="",
        )

    def save_pixel_values(
        self, output_location: pathlib.Path, channel_names_in: str | None = None, channel_names_out: str | None = None
    ) -> None:
        """Save raw, transformed and selected bands reference pixels to csv files."""
        if self.color_values_raw is not None:
            raw = output_location.joinpath("pixel_values/raw.csv")
            self.save_pixel_values_to_file(raw, self.color_values_raw, channel_names_in)
        if self.color_values_transformed is not None:
            transformed = output_location.joinpath("pixel_values/transformed.csv")
            self.save_pixel_values_to_file(transformed, self.color_values_transformed, channel_names_out)
        selected = output_location.joinpath("pixel_values/selected.csv")
        self.save_pixel_values_to_file(selected, self.color_values, channel_names_out, raw=False)

    @abstractmethod
    def calculate_statistics(self) -> None:
        """Calculate the necessary statistics for performing the distance calculation, i.e. the covariance and average or similar.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def calculate_distance(self, image: NDArray[Any]) -> NDArray[Any]:
        """Calculate the color distance for each pixel in the image to the reference.
        Subclasses must implement this and call super() as the first thing to apply band selection and transforms.
        """
        if self.transform is not None:
            image = self.transform.transform(image)
        image = image[self.bands_to_use, :, :]
        return image

    @abstractmethod
    def show_statistics(self) -> None:
        """Print the statistics to screen.
        Subclasses must implement this and print the calculated statistics to screen.
        """
        pass


class MahalanobisDistance(BaseDistance):
    """
    A multivariate normal distribution used to describe the color of a set of
    pixels.
    """

    def calculate_statistics(self) -> None:
        """Calculate covariance and average."""
        self.covariance: NDArray[Any] = np.cov(self.color_values)
        self.average = np.average(self.color_values, axis=1)

    def calculate_distance(self, image: NDArray[Any]) -> NDArray[Any]:
        """
        Calculate the color distance using mahalanobis
        for each pixel in the image to the reference.
        """
        image = super().calculate_distance(image)
        pixels = np.reshape(image, (image.shape[0], -1)).transpose()
        inv_cov = np.linalg.inv(self.covariance)
        diff = pixels - self.average
        modified_dot_product = diff * (diff @ inv_cov)
        distance = np.sum(modified_dot_product, axis=1)
        distance = np.sqrt(distance)
        distance_image = np.reshape(distance, (1, image.shape[1], image.shape[2]))
        return distance_image

    def show_statistics(self) -> None:
        """Print the statistics to screen."""
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
    """

    def __init__(
        self,
        *,
        n_components: int,
        reference_pixels: NDArray[Any],
        bands_to_use: tuple[int, ...] | list[int] | None = None,
        transform: BaseTransform | None = None,
    ):
        self.n_components = n_components
        super().__init__(reference_pixels=reference_pixels, bands_to_use=bands_to_use, transform=transform)

    @classmethod
    def from_image_annotation(  # type: ignore[override]
        cls,
        *,
        n_components: int,
        reference: pathlib.Path,
        annotated: pathlib.Path,
        bands_to_use: tuple[int, ...] | list[int] | None = None,
        alpha_channel: int | None = -1,
        transform: BaseTransform | None = None,
        **kwargs: Any,
    ) -> GaussianMixtureModelDistance:
        return super().from_image_annotation(  # type: ignore[return-value]
            reference=reference,
            annotated=annotated,
            bands_to_use=bands_to_use,
            alpha_channel=alpha_channel,
            transform=transform,
            n_components=n_components,
            **kwargs,
        )

    @classmethod
    def from_pixel_values(  # type: ignore[override]
        cls,
        *,
        n_components: int,
        pixel_values: NDArray[Any],
        bands_to_use: tuple[int, ...] | list[int] | None = None,
        alpha_channel: int | None = -1,
        transform: BaseTransform | None = None,
        **kwargs: Any,
    ) -> GaussianMixtureModelDistance:
        return super().from_pixel_values(  # type: ignore[return-value]
            pixel_values=pixel_values,
            bands_to_use=bands_to_use,
            alpha_channel=alpha_channel,
            transform=transform,
            n_components=n_components,
            **kwargs,
        )

    def calculate_statistics(self) -> None:
        """Calculate covariance and average."""
        self.gmm = mixture.GaussianMixture(n_components=self.n_components, covariance_type="full")
        self.gmm.fit(self.color_values.transpose())
        self.average = self.gmm.means_
        self.covariance = self.gmm.covariances_
        self.max_score = np.max(self.gmm.score_samples(self.color_values.transpose()))

    def calculate_distance(self, image: NDArray[Any]) -> NDArray[Any]:
        """
        Calculate the color distance using a gaussian mixture model
        for each pixel in the image to the reference.
        """
        image = super().calculate_distance(image)
        pixels = np.reshape(image, (image.shape[0], -1)).transpose()
        loglikelihood = self.gmm.score_samples(pixels)
        distance = np.sqrt(np.maximum(-loglikelihood + self.max_score, 0))
        distance_image = np.reshape(distance, (1, image.shape[1], image.shape[2]))
        return distance_image

    def show_statistics(self) -> None:
        """Print the statistics to screen."""
        print("GMM")
        print(self.gmm)
        print(self.average)
        print(self.covariance)
