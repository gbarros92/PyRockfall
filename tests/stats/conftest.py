import numpy as np
import pytest

import pyrockfall.stats as stats

SEED = 12345


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def seed():
    return SEED


_CONTINUOUS_FACTORIES = [
    ("Normal", lambda: stats.Normal(0.0, 1.0)),
    ("Uniform", lambda: stats.Uniform(2.0, 5.0)),
    ("Triangular", lambda: stats.Triangular(0.0, 1.0, 3.0)),
    ("Beta", lambda: stats.Beta(2.0, 3.0)),
    ("Exponential", lambda: stats.Exponential(2.0)),
    ("Lognormal", lambda: stats.Lognormal(0.0, 0.5)),
    ("Gamma", lambda: stats.Gamma(2.0, 3.0)),
]


@pytest.fixture(
    params=[factory for _, factory in _CONTINUOUS_FACTORIES],
    ids=[name for name, _ in _CONTINUOUS_FACTORIES],
)
def any_continuous_dist(request):
    return request.param()
