Probability Distributions
=========================

The v2 distributions module provides tools for representing different types
of uncertain inputs.

Categorical Distribution
------------------------

``CategoricalDistribution`` represents a variable that can take one of
several discrete categories.

It is useful when an input is selected from a set of possible categories,
rather than sampled from a continuous range.

Continuous Distribution
-----------------------

``ContinuousDistribution`` represents a numeric variable over a continuous
range.

It can be used to represent uncertain quantities such as flood depth,
velocity, or duration when a suitable distribution is specified.

Empirical Distribution
----------------------

``EmpiricalDistribution`` represents a distribution based on observed
sample data.

It can be useful when an input is modeled using an existing dataset rather
than assuming a theoretical probability distribution.

Module Reference
----------------

For constructor arguments, supported distribution types, and sampling
behavior, see the API reference for
:mod:`pyinsyde.v2.distributions`.