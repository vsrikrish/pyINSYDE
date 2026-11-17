Missing Input Handling
======================

The v2 missing inputs module provides functions for identifying and
handling missing building characteristics.

Identifying Missing Inputs
--------------------------

``find_missing_inputs()`` checks a building data dictionary and identifies
fields that have not been provided.

.. code-block:: python

   from pyinsyde.v2 import find_missing_inputs

   missing = find_missing_inputs(building_data)

The function returns a list of missing input fields.

Filling Missing Inputs
----------------------

``fill_missing_inputs()`` provides a way to fill missing building data
using the functionality implemented in the v2 module.

.. code-block:: python

   from pyinsyde.v2 import fill_missing_inputs

   completed_data = fill_missing_inputs(building_data)

For the available arguments and the specific behavior of each function,
see the API reference for :mod:`pyinsyde.v2.missing_inputs`.

Why Missing Input Handling Matters
----------------------------------

Building datasets may contain incomplete records. Identifying and
handling missing values helps prepare building characteristics for
subsequent damage calculations.