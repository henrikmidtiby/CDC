import numpy as np
from sklearn import mixture


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
        self.gmm = mixture.GaussianMixture(n_components=self.n_components, covariance_type="full")
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
