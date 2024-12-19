Contributing
============

Thank you for your interest in contributing to *OCDC* and we welcome all pull request. To get set for development on *OCDC* see the following.

Development uses pre-commit for code linting and formatting. To setup development with pre-commit follow these steps after cloning the repository:

Create a virtual environment with python:

.. code-block:: shell

    python -m venv venv

Activate virtual environment:

.. code-block:: shell

    source venv/bin/activate

Install *OCDC* python package as editable with the development dependencies:

.. code-block:: shell

    pip install -e .[dev]

Install pre-commit hooks

.. code-block:: shell

    pre-commit install

You are now ready to contribute.
