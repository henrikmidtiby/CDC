# Howto - Install Orthomosaic Color Distance Calculator
The **Orthomosaic Color Distance Calculator** tool is written in python and relies on a set of python packages (opencv-python, numpy, rasterio among others). To handle installation of the required python libraries the **pip** tool can be used.

To install the tool, enter the following commands on the command line.
```bash
git clone https://github.com/henrikmidtiby/ColorBasedSegmenterForOrthomosaics.git
cd ColorBasedSegmenterForOrthomosaics
```

To create a virtual environment and install the required packages, enter the following commands on the command line.
```bash
python3 -m venv venv
venv/Scripts/activate
pip3 install -r requirements.txt
```

To run the script, enter the following commands on the command line.
```bash
./venv/Scripts/python.exe color_based_segmenter.py Tests/rodsvingel/input_data/2023-04-03_Rodsvingel_1._ars_Wagner_JSJ_2_ORTHO.tif Tests/rodsvingel/input_data/original.png Tests/rodsvingel/input_data/annotated.png --output_tile_location Tests/rodsvingel/tiles --tile_size 500
```

To remove the virtual environment, enter the following commands on the command line.
```bash
deactivate
rm -r venv
rm -r __pycache__
```