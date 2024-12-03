# Reference - Calculating distances to reference color

In this document the central part of the color distance calculation is explained. The calculation consists of two steps
1. establish a reference color model
2. calculating distances to the reference color model


## Establishing a reference color model
The standard color distance that is used in **Orthomosaic Color Distance Calculator**, is the Mahalanobis distance. The Mahalanobis distance is a measure of the distance from a point to a multivariate normal distribution.
The multivariate normal distribution is used to describe the distribution of color values from a set of annotated pixels. The multivariate normal distribution is described by the mean value $\vec{\mu}$ and the covariance matrix $\Sigma$ calculated from the $(R,G,B)$ color values of the sampled pixels. The mean value $\vec{\mu}$ is a $<3 \times 1>$ column vector and the covariance matrix is a $<3 \times 3>$ matrix.

## Calculating distance to the color model
To calculate the color distance using the Mahalanobis distance, the following equation is used:
$$\sqrt{\left( \vec{x} - \vec{\mu} \right)^T \cdot \Sigma^{-1} \cdot \left( \vec{x} - \vec{\mu} \right)}$$
where $\vec{x}$ is the new color value $\vec{x}$, $\vec{\mu}$ the mean color value and $\Sigma$ the covariance matrix.
