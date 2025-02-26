CDC - Color Distance Calculator
============================================

This tool, *CDC* makes it possible to calculate a distance to a reference color for each pixel in georeferenced orthomosaic.

*CDC* can be a useful tool if you want to locate objects in an image / orthomosaic with a certain color.
The tool can be used to go from this input image:

.. image:: _static/pumpkins_example/crop_from_orthomosaic.png

To this output image:

.. image:: _static/pumpkins_example/color_distance_crop.png

To learn more about the tool, take a look at the :doc:`Tutorial - Segment pumpkins in RGB orthomosaic <tutorials_guides>`

Installation
------------

*CDC* is a python package and can be installed with pip:

.. code-block:: shell

   pip install CDC

See :doc:`Installation <installation>` for more advanced methods of installation.

Acknowledgement
---------------

the *CDC* tool was developed by SDU UAS Center as part of the project *Præcisionsfrøavl*, that was supported by the `Green Development and Demonstration Programme (GUDP) <https://gudp.lbst.dk/>`_ and `Frøafgiftsfonden <https://froeafgiftsfonden.dk/>`_ both from Denmark.

Index
-----

.. toctree::
   :maxdepth: 2

   installation
   tutorials_guides
   reference
   CLI
   contributing
   calculate_dist
