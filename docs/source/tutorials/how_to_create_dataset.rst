Create dataset guide
====================

*CDC* Needs an annotated image where the reference pixels from which the distance is calculated. In this guide we will show how to extract a section of a orthomosaic and how to annotate the section.

This guide assumes we have an orthomosaic with the name **ortho.tif**.

Crop orthomosaic
----------------

To extract a section from the orthomosaic we will use `GDAL <https://gdal.org/en/stable/index.html>`_ but other software such as `QGIS <https://www.qgis.org/>`_ can also be used.

The following command will extract a section from the orthomosaic:

.. code-block:: shell

    gdal_translate ortho.tif ref.tif -srcwin 7000 6000 500 400

- ortho.tif is the orthomosaic from which the section is taken.
- ref.tif is what to save the section as.
- 7000 is the x coordinate for the top left corner of the section.
- 6000 is the y coordinate for the top left corner of the section.
- 500 is the width in the x direction of the section.
- 400 is the width in the y direction of the section.

Here it is important to choose a section where the desired color from which to calculate the distance is present and is a good approximation of the color for the whole orthomosaic. It can be necessary to play around with the position and size of the section to get a good reference.

To get a PNG image on which the annotations can be made the following command can be used with the same section position and size as above:

.. code-block:: shell

    gdal_translate ortho.tif mask.png -srcwin 7000 6000 500 400 -scale

If the orthomosaic is multispectral and not just RGB we also need to specify which image channels to use for the PNG image. This can be done with the following command:

.. code-block:: shell

    gdal_translate ortho.tif mask.png -srcwin 7000 6000 500 400 -scale -b 1 -b 2 -b 3

where *-b number* is the channel to select. Here we select the first 3 channels.

Mask Annotation
---------------

To annotate the mask we will use `GIMP <https://www.gimp.org/>`_, but another image manipulations software can also be used.

- Open the mask.png we created earlier in GIMP.
- Set the color to red. E.g. RGB (255, 0, 0).
- Select the pencil tool and adjust size to preference.
- paint on top of the desired pixels to include in the reference color.
- Not all pixels of the desired color have to be painted over but at least 100 pixels is necessary, but more is better.
- Export the image (File -> Export as). We override the mask.png.

We now have all we need to calculate the color distance with *CDC*

- The orthomosaic (ortho.tif).
- reference image (ref.tif).
- annotated image (mask.png).
