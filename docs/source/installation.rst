Installation
============

CDC is a python package and can be installed with pip:

.. code-block:: shell

   pip install CDC

It have the following main dependencies:

* matplotlib
* numpy
* rasterio
* scikit-learn
* tqdm

If the latest version is desired *CDC* can also be installed directly from git by cloning the repository and in the project folder running:

.. code-block:: shell

   pip install .

If you want changes in the code to be reflected install it as an editable with:

.. code-block:: shell

   pip install -e .

All of these methods can also be used inside a virtual environment:

.. code-block:: shell

   python -m venv venv
   source venv/bin/activate
   pip install CDC
