Running multi-threaded and handling warnings
============================================

When using the CLI to process an orthomosaic the program will as default run as multi-threaded. Because of a open bug in numpy (https://github.com/numpy/numpy/issues/27989) calculating the color distance which is using numpy matmul the program will crash on windows.

To avoid this problem we use a lock when doing the color distance calculation. This solve the problem, but at the cost of a small performance penalty. Even with the lock we still see a large improvement in performance compared to running as a single thread.

Another problem solved by introducing the lock on color distance calculation was numpy giving ``RuntimeWarnings`` on invalid inputs in ``np.sqrt`` leading to invalid data in the generated color distance orthomosaic. The cause of this is unknown, but is assumed to originate from simultaneously reading from the same array when calling ``calculate_distance`` from the ColorModel class.

Since RuntimeWarnings can lead to invalid output, and users are not expected to react to warnings, all warnings will be handled as errors. If a specific warning needs to be handled differently this can be done by setting ``filterwarnings`` specifically for that warning in ``__init__.py``. For example:

.. code-block:: python

    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
