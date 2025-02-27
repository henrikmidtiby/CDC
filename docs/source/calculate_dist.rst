Reference - Calculate distances to reference Color
==================================================

In this document the central part of the color distance calculation is explained. The calculation consists of two steps

1. establish a reference color model
2. calculating distances to the reference color model


Establishing a reference color model
------------------------------------

The standard color distance that is used in *CDC*, is the Mahalanobis distance [#Mahalanobis]_. The Mahalanobis distance is a measure of the distance from a point to a multivariate normal distribution.

The multivariate normal distribution is used to describe the distribution of color values from a set of annotated pixels. The multivariate normal distribution is described by the mean value :math:`\vec{\mu}` and the covariance matrix :math:`S` calculated from the :math:`(R,G,B)` color values of the sampled pixels. The mean value :math:`\vec{\mu}` is a :math:`<3 \times 1>` column vector and the covariance matrix is a :math:`<3 \times 3>` matrix.

Calculating distance to the color model using Mahalanobis
---------------------------------------------------------

To calculate the color distance using the Mahalanobis distance, the following equation is used:

.. math:: \sqrt{\left( \vec{x} - \vec{\mu} \right)^T \cdot S^{-1} \cdot \left( \vec{x} - \vec{\mu} \right)}

where :math:`\vec{x}` is the new color value, :math:`\vec{\mu}` the mean color value and :math:`S` the covariance matrix.


Calculating distance to the color model using Gaussian Mixture Model
--------------------------------------------------------------------

To calculate the color distance using the Gaussian Mixture Model [#gmm]_, the following equation is used to convert the loglikelihood to a distance:

.. math:: \sqrt{\max\{-\left(L-L_{max}\right),0\}}

Where :math:`L` is the loglikelihood from the Gaussian Mixture Model and :math:`L_{max}` is the maximum loglikelihood of the all the annotated pixels. :math:`L_{max}` is used as an approximation of the global maximum of the loglikelihood and the :math:`\max` function is to make sure value under the square root cannot be negative. The subtraction of :math:`L_{max}` is to achieve a minimum distance close to zero when the values are close to the annotated pixels.

.. rubric:: Footnotes

.. [#Mahalanobis] https://en.wikipedia.org/wiki/Mahalanobis_distance
.. [#gmm] https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
