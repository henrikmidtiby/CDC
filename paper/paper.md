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

- precision agriculture
- improved segmentation
- multispectral
- able to handle large orthomosaic by tiling.

A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work.

A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline.

Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it.

# Acknowledgements

the OCDC tool was developed by SDU UAS Center as part of the project Præcisionsfrøavl, that was supported by the [Green Development and Demonstration Programme (GUDP)](https://gudp.lbst.dk/) and [Frøafgiftsfonden](https://froeafgiftsfonden.dk/) both from Denmark.

# References

Notes: (to be deleted)

The paper should be between 250-1000 words

review_checklist:

- Summary: Has a clear description of the high-level functionality and purpose of the software for a diverse, non-specialist audience been provided?
- A statement of need: Does the paper have a section titled ‘Statement of need’ that clearly states what problems the software is designed to solve, who the target audience is, and its relation to other work?
- State of the field: Do the authors describe how this software compares to other commonly-used packages?
- Quality of writing: Is the paper well written (i.e., it does not require editing for structure, language, or writing quality)?
- References: Is the list of references complete, and is everything cited appropriately that should be cited (e.g., papers, datasets, software)? Do references in the text use the proper citation syntax?
