---
title: 'CDC: Color Distance Calculator'
Tags:
  - Python
  - Computer Vision
  - Orthomosaic
  - UAV
  - Image Processing
  - Color Distance
authors:
  - name: Henrik Skov Midtiby
    orcid: 0000-0002-3310-5680
    affiliation: '1'
  - name: Henrik Dyrberg Egemose
    corresponding: true
    orcid: 0000-0002-6765-8216
    affiliation: '1'
  - name: Søren Vad Iversen
    affiliation: '1'
  - name: Rasmus Storm
    affiliation: '1'
affiliations:
  - index: 1
    name: The Mærsk Mc-Kinney Møller Institute, University of Southern Denmark
date: 30 January 2025
bibliography: paper.bib
---

# Summary

The Color Distance Calculator (CDC) is an open-source python
package designed to calculate a color distance image, a gray scale image
with color distances from all pixels in the input image to a reference color.
It is specifically tailored for handling large orthomosaics and multispectral data.
By providing CDC with reference pixels, it calculates the distance using
either the Mahalanobis distance or a Gaussian Mixture Model for all pixels
in the orthomosaic.
Figure \ref{pumpkins} shows a small section of a pumpkin field and the corresponding color distance image.
CDC's main functions are exposed through a command-line interface, where
providing an orthomosaic, reference image, and a mask will output a new
orthomosaic with the color distances.
The Python package also allows for using CDC as a library for more complex tasks.

![Small section of pumpkins field (left) color distance image of pumpkins field (right) \label{fig:pumpkins}](pumkpiks_figure.png)


# Statement of need

A common task in Precision Agriculture is to segment an orthomosaic into
different regions based on the information in the orthomosaic.
The regions can represent areas with healthy vegetation or areas with
unwanted vegetation.
The classic approach is to use the excess green (ExG) color index to assess
whether the current pixel is green enough to be considered healthy vegetation.
Such an approach based on a hardcoded, rule (ExG and a threshold) is only
suitable for a limited number of cases.

Given enough training data, it is possible to train convolutional neural
networks (CNN's) for segmenting arbitrary objects in images
[@Ronneberger2015Unet].
However, obtaining enough annotated training data can be difficult.

A more flexible approach is to use a small set of pixels to determine a reference
color and then calculate the distance for all pixels in the input image
to that reference color.
We have successfully used this approach in several cases, including the following:

- detect healthy crop plants in a grass seed field
- locating thistles in a grass seed field
- counting pumpkins in a pumpkin field [@midtiby2022]

We propose CDC for segmenting large multispectral orthomosaics by calculating
the color distance to a set of reference pixels.
The Output of CDC is a grayscale orthomosaic which can easily be threshold to
achieve a black and white segmentation.

CDC is developed with Agriculture uses in mind, but can easily be applied to
other domains as is or by utilizing the library for custom needs.

# Background and methods

In Precision Agriculture a common application is to assess the vegetation
health by using Remote Sensing techniques and image analytics.
The most applied Remote Sensing techniques is arial monitoring, where images
from satellites, manned aircraft, and Unmanned Aerial Vehicles (UAVs) are
captured [@matese2015].
The use of UAVs, also known as drones, has seen a large increase in recent
years as they provide high-quality images at a more affordable cost
than satellites and manned aircraft [@pareview2020].

UAVs can carry various kinds of cameras, such as multispectral and hyperspectral,
along with normal RGB cameras, thereby acquiring aerial images that can be
used to extract vegetation indices.
Vegetation indices such as the Normalized Difference Vegetation Index (NDVI) can
be interpreted by farmers to monitor the crops variability and stress[@xue2017].
Individual images from the UAV normally only cover a small part of the field,
but to get a overview of the whole field, the images are stitched together in
software like OpenDroneMap, Metashape, or Pix4D together with
Geographical Information Systems (GIS) information, creating a large
georeferenced orthomosaic.

In CDC the Mahalanobis distance in the RGB color space is used as the default
color distance. The Mahalanobis distance is defined as

$$
\sqrt{\left( \vec{x} - \vec{\mu} \right)^T \cdot S^{-1} \cdot \left( \vec{x} - \vec{\mu} \right)}
$$

where $\vec{x}$ is the color value, $\vec{\mu}$ the mean color value and $S$ the covariance matrix.
The parameters $\vec{\mu}$ and $S$ are determined from a set of training pixels.


# Acknowledgements

The CDC tool was developed by SDU UAS Center as part of the project
Præcisionsfrøavl, that was supported by the
[Green Development and Demonstration Programme (GUDP)](https://gudp.lbst.dk/) and
[Frøafgiftsfonden](https://froeafgiftsfonden.dk/) both from Denmark.

# References
