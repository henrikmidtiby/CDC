---
title: 'OCDC: Orthomosaic Color Distance Calculator'
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

OCDC is a open-source python package for calculating color distances in images. It is specifically made for handling large orthomosaics and multispectral data. By providing OCDC with reference pixels it calculates the distance using Mahalanobis distance or a Gaussian Mixture Model for all pixels in the orthomosaic. OCDCs main function are exposed as a command line interface where providing an orthomosaic, reference image and a mask will output a new orthomosaic with the color distances. The python package also allow for using OCDC as a library for more complex tasks.

# Statement of need

In Precision Agriculture the most common application is to assess the vegetation health by using Remote Sensing techniques and image analytics. The most applied Remote Sensing techniques is arial monitoring where images from satellites, manned aircraft and Unmanned Aerial Vehicles (UAVs) are captured [@matese2015]. The use of UAVs also known as drones have seen a large increase in resent years as it is a more economical solution and are capable of providing high-quality images then satellites and manned aircraft [@pareview2020].

UAVs can carry various kinds of cameras such as multispectral and hyperspectral along with normal RGB cameras, thereby acquiring aerial images that can be used to extract vegetation indices. Vegetation indices such as Normalized Difference Vegetation Index (NDVI) can be interpreted by farmers to monitor the crops variability and stress[@xue2017]. Individual images from the UAV normally only covers a small part of the field, but to get a overview of the whole field the images are stitch together in software like OpenDroneMap and Pix4D together with Geographical Information Systems (GIS) information creating a large georeferenced orthomosaic.

More advanced Precision Agriculture applications like yield estimation [@midtiby2022] and crop row detection requires segmentation in order to be able to tell what is crop and what is background. This can be done with deep neural networks [@PANG2020], but comes with the cost of needing training images and are often crop specific, so entry cost in applying deep neural networks can often be high.

We purpose OCDC for segmenting large multispectral orthomosaics by calculating the color distance to a set of reference pixels. The Output of OCDC is a grayscale orthomosaic which can easily be threshold to achieve a black and white segmentation.

OCDC is developed with Agriculture uses in mind, but can easily be applied to other domains as is or by utilizing the library for custom needs.


# Acknowledgements

the OCDC tool was developed by SDU UAS Center as part of the project Præcisionsfrøavl, that was supported by the [Green Development and Demonstration Programme (GUDP)](https://gudp.lbst.dk/) and [Frøafgiftsfonden](https://froeafgiftsfonden.dk/) both from Denmark.

# References
