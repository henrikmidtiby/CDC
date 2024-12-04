# OCDC - Orthomosaic Color Distance Calculator

OCDC can be a useful tool if you want to locate objects in an image / orthomosaic with a certain color. The tool can be used to go from this input image

![Image](documentation/pumpkins_example/crop_from_orthomosaic.png)

To this output image

![Image](documentation/pumpkins_example/color_distance_crop.png)

To learn more about the tool, take a look at the tutorial.
* [Tutorial - Segment pumpkins in RGB orthomosaic](Tutorial_Segment_pumpkins_in_rgb_orthomosaic.md)

## Installation

OCDC is a python package and can be installed with pip. First download the code from a release (#to-do insert link to release) or get the latest by cloning the repository.

```
pip install .
```

If installing from at different directory replace `.` with the path to the code.

## Quick start

### How to make a reference image and mask

#todo

### Run OCDC

To run OCDC on an orthomosaic, run the following in a terminal window:

```
OCDC path/to/orthomosaic path/to/reference_image path/to/mask_image
```

See `OCDC --help` for more information.

#todo expand on all the parameters for OCDC

## Development

Clone the repositiry and pip install as editable with the development dependencies:

```
pip install -e .[dev]
```

Install pre-commit:

```
pre-commit install
```

This will ensure during development that each of your commits is properly formatted against our linter and formatters.

You are now ready to work on OCDC.

## Acknowledgement

#todo
