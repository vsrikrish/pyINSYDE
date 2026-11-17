Uncertainty Analysis
====================

The v2 uncertainty module provides tools for evaluating how uncertainty
in model inputs can affect estimated building damage.

Sampling Damage
---------------

``sample_damage()`` runs damage calculations using sampled inputs.

This allows multiple damage estimates to be generated under different
input scenarios.

.. code-block:: python

   from pyinsyde.v2 import sample_damage

   damages = sample_damage(...)

The exact arguments depend on the sampling configuration and building
information. See the API reference for details.

Summarizing Damage
------------------

``summarize_damage()`` summarizes a collection of damage estimates.

.. code-block:: python

   from pyinsyde.v2 import summarize_damage

   summary = summarize_damage(damages)

This can help describe the distribution of estimated damage across
multiple simulations.

Module Reference
----------------

For supported arguments, sampling behavior, and summary statistics,
see :mod:`pyinsyde.v2.uncertainty`.