# Optional arguments for run_segmenter.py
The **run_segmenter.py** script has 3 positinal arguments:
``` bash 
$ python3 color-based-imagesegmentation/run_segmenter.py [orthomosaic] [reference image] [pixel mask]
```
Beyond this it can take number of optional arguments for increased usability.



# --method (methods of segmentation)
The script contains different methods for creating the resulting gray scale distance image from a setof reference pixels. Each of the methods based on an underlying model for the probability of the apixels being in the category of interest. Which of these color-models to use can be chosen using the
```bash
--method [name of colormodel].
```
The following colormodels are implemented:
- 'gmm'
-  'mahalanobis'
The default is using 'mahalanobis'.




An exampole of the use of **--method** can be seen below, segmenting using the Gausian mixture
model
```bash
$ python3 color-based-imagesegmentation/run_segmenter.py [orthomosaic] [reference image] [
pixel mask] --method gmm
```
Some color-models might need an extra argument. These can be given using the **--params [extra argument]**.

As an example to segment using a Gaussian mixture model, the number of clusters is a free parameter,if this is not specified the script will auitomaticly run assuming 2 clusters. The **--param** option allows the user to change this to fx. 3:
```bash
$ python3 color-based-imagesegmentation/run_segmenter.py [orthomosaic] [reference image] [
pixel mask] --method gmm --param 3
```

Everthing pertaining the color-models are located in the segmenter/colormodels.py.



# --bands_to_use
This option allows the user to select which color bands to use for the segmentation. It must be followedby a list of integers indicating which bands are of interest, this is zero indexed. The example belowruns the segmentation only using band a and b.
```bash
$ python3 color-based-imagesegmentation/run_segmenter.py [orthomosaic] [reference image] [
pixel mask] --bands_to_use a b
```
# --scale_factor (scaling of the output images)
The distance images created by the script are saved using 8 bits per pixel and therefore with eachpixel an integer in the range from 0-255. To ensure the a good use of the available range, the script multiplies the output values of the color-models with a scale factor α. The value of α can be changed using:
``` bash
--scale_factor [value]
```
Where value is some positive float. The default value is α = 5

# --notiling (for segmenting smaller images)
Though the program is developed with segmentation of large orthomosaics, therefore sectioning the
image of interest into smaller tiles, in some cases the image of interest is small and no tiling is needed and not preferred. To stop the program form tiling the image one can use the **--notiling** option:
```bash
$ python3 color-based-imagesegmentation/run_segmenter.py [orthomosaic] [reference image] [
pixel mask] --notiling
```
# --tile_size
Changes the size of the tiles generated segmented
```bash
--tile_size [size of tile in pixels]
```
The default tile size is 3000 pixels, generating 3000 × 3000 pixel tiles.
Example:
```bash
$ python3 color-based-imagesegmentation/run_segmenter.py [orthomosaic] [reference image] [
pixel mask] --tile_size 500
```
# --output_tile_location
To change the output location of the distance images one can use the 
```bash
--output_tile_location [path to output location]
```

The default output location is the folder output/mahal, if the folder doesnt exist before runtime this will be created in the current working directory.
Example:
```bash
$ python3 color-based-imagesegmentation/run_segmenter.py [orthomosaic] [reference image] [
pixel mask] --output_tile_location ’output/mahal’
```

# --ref_pixel_value
The program saves the values of the reference pixels used in the segmentation as a **.txt** as a default this is named **Referencepixels.txt**. If a file with the same name already exists in the current workingdirectory this will be owerwritten. The path and filename can be changed using:
```bash
--ref_pixel_filename [new name]
```

# --ref_method
To obtain the reference pixels for building the color-models used for segmentation, the program uses a reference image and a mask indicating some of the pixels in the class of interest in that picture. 
As a default this mask must be given as a *.tif* file with 1 band, with the pixel value 255 where the reference image has the class of interest, and 0 elsewhere.

The script supports alternate methods with the 
```bash
--ref_method [name of method used for mask]
```

The only alternate mask option implemented is the **’jpg’** method allowing the user to annotate the reference image in fx. GIMP(or another image editor) using the red colour and exporting it as an
**.jpg**. An example of using this method is shown below:

```bash
$ python3 color-based-imagesegmentation/run_segmenter.py [orthomosaic] [reference image]
pixel_mask.jpg --ref_method jpg
```