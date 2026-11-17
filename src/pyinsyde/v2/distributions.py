import numpy as np
from scipy.stats import truncnorm


class CategoricalDistribution:
    """Probability distribution for categorical values."""

    def __init__(self, values, probabilities):
        if len(values) != len(probabilities):
            raise ValueError(
                "Values and probabilities must have the same length."
            )

        if len(values) == 0:
            raise ValueError("At least one value is required.")

        try:
            probabilities = np.asarray(probabilities, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Probabilities must be numeric.") from exc

        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Probabilities must be finite.")

        if np.any(probabilities < 0):
            raise ValueError(
                "Probabilities must be non-negative."
            )

        if not np.isclose(probabilities.sum(), 1.0, atol=1e-9):
            raise ValueError("Probabilities must sum to 1.")

        self.values = list(values)
        self.probabilities = probabilities.tolist()

    def sample(self, rng=None):
        """Draw one value from the distribution."""
        if rng is None:
            rng = np.random.default_rng()

        # Choose an index to preserve the original value's type.
        index = rng.choice(
            len(self.values),
            p=self.probabilities,
        )
        return self.values[index]


class ContinuousDistribution:
    """Normal distribution for continuous values, with optional bounds."""

    def __init__(self, mean, sd, lower=None, upper=None):
        try:
            mean = float(mean)
            sd = float(sd)
            lower = None if lower is None else float(lower)
            upper = None if upper is None else float(upper)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Mean, standard deviation, and bounds must be numeric."
            ) from exc

        if not np.isfinite(mean):
            raise ValueError("Mean must be finite.")

        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(
                "Standard deviation must be finite and positive."
            )

        if lower is not None and not np.isfinite(lower):
            raise ValueError("Lower bound must be finite.")

        if upper is not None and not np.isfinite(upper):
            raise ValueError("Upper bound must be finite.")

        if lower is not None and upper is not None and lower > upper:
            raise ValueError("Lower bound cannot exceed upper bound.")

        self.mean = mean
        self.sd = sd
        self.lower = lower
        self.upper = upper

    def sample(self, rng=None):
        """Draw one value from the normal distribution within its bounds."""
        if rng is None:
            rng = np.random.default_rng()

        # Equal bounds define a distribution concentrated at one value.
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower == self.upper
        ):
            return self.lower

        # Without bounds, use the standard normal sampler.
        if self.lower is None and self.upper is None:
            return rng.normal(self.mean, self.sd)

        # Convert the bounds to standard-normal units.
        a = (
            -np.inf
            if self.lower is None
            else (self.lower - self.mean) / self.sd
        )
        b = (
            np.inf
            if self.upper is None
            else (self.upper - self.mean) / self.sd
        )

        return truncnorm.rvs(
            a,
            b,
            loc=self.mean,
            scale=self.sd,
            random_state=rng,
        )


class EmpiricalDistribution:
    """Probability distribution based on observed data."""

    def __init__(self, samples):
        if len(samples) == 0:
            raise ValueError("At least one sample is required.")

        self.samples = list(samples)

    def sample(self, rng=None):
        """Draw one value from the observed data."""
        if rng is None:
            rng = np.random.default_rng()

        # Choose an index to preserve the original sample's type.
        index = rng.choice(len(self.samples))
        return self.samples[index]
