

.PHONY: run clean

.DEFAULT_GOAL := run

VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

run: $(VENV)/bin/activate
	$(PYTHON) color_based_segmenter.py Tests/rødsvingel/input_data/2023-04-03_Rødsvingel_1._års_Wagner_JSJ_2_ORTHO.tif Tests/rødsvingel/input_data/original.png Tests/rødsvingel/input_data/annotated.png --output_tile_location Tests/rødsvingel/tiles --tile_size 500

clean: 
	rm -rf $(VENV)
	rm -rf __pycache__
