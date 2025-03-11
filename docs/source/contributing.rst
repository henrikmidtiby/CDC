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

Creating Github Release
-----------------------

When a new release is desired from the commits to the master branch, the following steps will create a new release and bump the version number:

* Change version number in :code:`src/CDC/__init__.py` and commit to master.
* Tag the commit with the version number: :code:`git tag vXX.XX.XX`.
* Push the changes to github: :code:`git push origin` (where origin is the name of github upstream).
* push the tag to github: :code:`git push origin tag vXX.XX.XX`.

This will start the github actions to create a new release and publish the code to PyPI together with generating the new documentation.
