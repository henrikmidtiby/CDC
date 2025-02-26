Contributing
============

Thank you for your interest in contributing to *CDC* and we welcome all pull request. To get set for development on *CDC* see the following.

Development uses pre-commit for code linting and formatting. To setup development with pre-commit follow these steps after cloning the repository:

Create a virtual environment with python:

.. code-block:: shell

    python -m venv venv

Activate virtual environment:

.. code-block:: shell

    source venv/bin/activate

Install *CDC* python package as editable with the development dependencies:

.. code-block:: shell

    pip install -e .[dev]

Install pre-commit hooks

.. code-block:: shell

    pre-commit install

You are now ready to contribute.

Running Test
------------

Test is automatically run when making a commit, but can also be run with:

.. code-block:: shell

    pytest

This will also generate a html coverage report in *test_coverage*.

Generating Documentation
------------------------

To generate this documentation, in the *docs* folder run:

.. code-block:: shell

    make html

This will generate html documentation in the *docs/build/html* folder.
