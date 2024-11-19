from abc import ABC, abstractmethod

import numpy as np
import rasterio
from sklearn import mixture


class ReferencePixels:
    def __init__(self, *, reference, annotated, bands_to_use, **kwargs):
        self.reference_image_filename = reference
        self.mask_filename = annotated
        self.bands_to_use = bands_to_use
        self.reference_image = None
        self.mask = None
        self.values = None
        self.initialize()

    def initialize(self):
        self.load_reference_image(self.reference_image_filename)
        self.load_mask(self.mask_filename)
        if self.bands_to_use is None:
            self.bands_to_use = tuple(range(self.reference_image.shape[0] - 1))
        self.generate_pixel_mask()
        self.show_statistics_of_pixel_mask()

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
        # fix header for csv file
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
    """Base class for all color distance models."""

    def __init__(self, **kwargs):
        self.reference_pixels = ReferencePixels(**kwargs)
        self.bands_to_use = self.reference_pixels.bands_to_use

    def initialize(self):
        self.calculate_statistics()
        self.show_statistics()

    def save_pixel_values(self, filename):
        self.reference_pixels.save_pixel_values_to_file(filename)

    @abstractmethod
    def calculate_statistics(self):
        pass

    @abstractmethod
    def calculate_distance(self, image):
        pass

    @abstractmethod
    def show_statistics(self):
        pass


class MahalanobisDistance(BaseDistance):
    """
    A multivariate normal distribution used to describe the color of a set of
    pixels.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.average = None
        self.covariance = None

    def calculate_statistics(self):
        self.covariance = np.cov(self.reference_pixels.values)
        self.average = np.average(self.reference_pixels.values, axis=1)

    def calculate_distance(self, image):
        """
        For all pixels in the image, calculate the Mahalanobis distance
        to the reference color.
        """
        pixels = np.reshape(image[self.bands_to_use, :, :], (len(self.bands_to_use), -1)).transpose()
        inv_cov = np.linalg.inv(self.covariance)
        diff = pixels - self.average
        modified_dot_product = diff * (diff @ inv_cov)
        distance = np.sum(modified_dot_product, axis=1)
        distance = np.sqrt(distance)
        distance_image = np.reshape(distance, (1, image.shape[1], image.shape[2]))
        return distance_image

    def show_statistics(self):
        print("Average color value of annotated pixels")
        print(self.average)
        print("Covariance matrix of the annotated pixels")
        print(self.covariance)


class GaussianMixtureModelDistance(BaseDistance):
    def __init__(self, n_components, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.gmm = None

    def calculate_statistics(self):
        self.gmm = mixture.GaussianMixture(n_components=self.n_components, covariance_type="full")
        self.gmm.fit(self.reference_pixels.values.transpose())

    def calculate_distance(self, image):
        """
        For all pixels in the image, calculate the distance to the
        reference color modelled as a Gaussian Mixture Model.
        """
        pixels = np.reshape(image[self.bands_to_use, :, :], (len(self.bands_to_use), -1)).transpose()
        distance = self.gmm.score_samples(pixels)
        distance_image = np.reshape(distance, (1, image.shape[1], image.shape[2]))
        return distance_image

    def show_statistics(self):
        print("GMM")
        print(self.gmm)
        print(self.gmm.means_)
        print(self.gmm.covariances_)
