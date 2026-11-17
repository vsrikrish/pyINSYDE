Input Validation
================

The v2 validation module checks flood hazard inputs before damage
calculations are performed.

The main validation function is ``validate_hazard_inputs()``.

Accepted Inputs
---------------

The function validates five hazard parameters:

+-----------+----------------------+-----------------------------+
| Parameter | Description          | Constraint                  |
+===========+======================+=============================+
| ``he``    | External water height| Finite and nonnegative      |
+-----------+----------------------+-----------------------------+
| ``v``     | Flow velocity        | Finite and nonnegative      |
+-----------+----------------------+-----------------------------+
| ``d``     | Flood duration       | Finite and nonnegative      |
+-----------+----------------------+-----------------------------+
| ``s``     | Sediment concentration| Finite and nonnegative     |
+-----------+----------------------+-----------------------------+
| ``q``     | Pollutant presence   | Boolean or numeric 0 or 1   |
+-----------+----------------------+-----------------------------+

Using the Validation Function
-----------------------------

You can call the validation function directly:

.. code-block:: python

   from pyinsyde.v2 import validate_hazard_inputs

   validate_hazard_inputs(
       he=3.0,
       v=0.5,
       d=24,
       s=0.05,
       q=0
   )

If the inputs are valid, the function completes without raising an
exception.

Error Handling
--------------

The validation function raises:

* ``TypeError`` when an input has an invalid type.
* ``ValueError`` when a numeric input is non-finite, negative, or outside
  the allowed binary values for ``q``.

Validation in Damage Calculations
---------------------------------

The v2 ``compute_damage()`` wrapper calls
``validate_hazard_inputs()`` before passing the inputs to the original
building damage calculation.

This means invalid hazard inputs are rejected before the original
calculation runs.

Calling ``Building.compute_damage()`` directly from the original
implementation bypasses this v2 validation wrapper.