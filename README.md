# CDC - Color Distance Calculator

CDC can be a useful tool if you want to locate objects in an image / orthomosaic with a certain color. The tool can be used to go from this input image

![Image](docs/source/_static/pumpkins_example/crop_from_orthomosaic.png)

To this output image

![Image](docs/source/_static/pumpkins_example/color_distance_crop.png)

To learn more about the tool, take a look at the tutorial.
* [Tutorial - Segment pumpkins in RGB orthomosaic](https://henrikmidtiby.github.io/CDC/tutorials/test_dataset_tutorial.html)

## Table of contents:

- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgement](#acknowledgement)

## Installation

CDC is a python package and can be installed with pip.

```
pip install CDC
```

For more advanced installation, please visit the [Documentation](https://henrikmidtiby.github.io/CDC/) for more information.

## Usage

For a tutorial of how to use CDC on a test dataset, please see [Tutorial](https://henrikmidtiby.github.io/CDC/tutorials_guides.html).

### How to make a reference image and mask

The easiest way to make a reference image is to use your preferred GIS image tool (like [QGIS](https://www.qgis.org/)) to extract a small region from the orthomosaic. To make the mask open the reference image in an image editor and use the pen tool to mark all the desired pixels with red ((255, 0, 0) in RGB).

### Run CDC

To run CDC on an orthomosaic, run the following in a terminal window:

```
CDC path/to/orthomosaic path/to/reference_image path/to/mask_image
```

Run `CDC --help` for more information or see the [Documentation](https://henrikmidtiby.github.io/CDC/).

## Documentation

For a full list of command line arguments see [CLI](https://henrikmidtiby.github.io/CDC/CLI.html). For a reference manual, please visit [Reference Manual](https://henrikmidtiby.github.io/CDC/reference.html)

## Contributing

For contribution guidelines, please see the [Documentation](https://henrikmidtiby.github.io/CDC/contributing.html).

## License

The software is licensed under the BSD-3-Clause license, see [License](LICENSE).

## Acknowledgement

the CDC tool was developed by SDU UAS Center as part of the project Præcisionsfrøavl, that was supported by the [Green Development and Demonstration Programme (GUDP)](https://gudp.lbst.dk/) and [Frøafgiftsfonden](https://froeafgiftsfonden.dk/) both from Denmark.
