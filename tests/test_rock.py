"""Tests for pyrockfall.Rock: mass/density as distributions, rvs/ppf moments.

Rock.mass and Rock.density are coerced via stats.asDistribution into
stats.Distribution objects. Rock.rvs(N) delegates to `<dist>.rvs(N)` (no
random_state parameter is exposed on Rock itself, so reproducibility across
calls relies on seeding the global NumPy legacy random state via
`np.random.seed(...)`, which scipy's frozen `rvs(random_state=None)` reads).
Rock.ppf(q) delegates to `<dist>.ppf(q[i])` for a (2, N) quantile array.

These tests define mass/density with distributions whose analytical mean and
variance are known, then verify that:
  * rvs() sample moments (large N) converge to the analytical moments, and
  * ppf() moments, computed via a deterministic midpoint quantile grid
    (numerical quadrature of the inverse-CDF), match the analytical moments
    closely (this is a much tighter/deterministic check than the rvs one,
    since it involves no randomness).
"""
import numpy as np
import pytest

import pyrockfall as pr
from pyrockfall import stats

MASS_MU, MASS_SIGMA = 50.0, 5.0  # Normal(mu, sigma)
DENSITY_LOWER, DENSITY_UPPER = 2500.0, 2700.0  # Uniform(lower, upper)


def make_rock():
    mass_dist = stats.Normal(MASS_MU, MASS_SIGMA)
    density_dist = stats.Uniform(DENSITY_LOWER, DENSITY_UPPER)
    return pr.Rock(name="TestRock", mass=mass_dist, density=density_dist)


# ---------------------------------------------------------------------------
# Construction / properties
# ---------------------------------------------------------------------------

def test_mass_and_density_are_the_assigned_distributions():
    rock = make_rock()
    assert isinstance(rock.mass, stats.Normal)
    assert isinstance(rock.density, stats.Uniform)
    assert rock.mass.mean() == pytest.approx(MASS_MU)
    assert rock.density.mean() == pytest.approx((DENSITY_LOWER + DENSITY_UPPER) / 2.0)


def test_default_mass_and_density_are_deterministic_zero():
    rock = pr.Rock(name="Empty")
    assert isinstance(rock.mass, stats.Deterministic)
    assert isinstance(rock.density, stats.Deterministic)
    assert rock.mass.value == 0.0
    assert rock.density.value == 0.0


def test_name_defaults_to_generated_group_name():
    rock = pr.Rock(mass=1.0, density=1.0)
    assert rock.name.startswith("Group ")


def test_scalar_mass_and_density_are_coerced_to_deterministic():
    rock = pr.Rock(name="Fixed", mass=12.5, density=2600.0)
    assert isinstance(rock.mass, stats.Deterministic)
    assert rock.mass.value == 12.5
    assert isinstance(rock.density, stats.Deterministic)
    assert rock.density.value == 2600.0


def test_mass_and_density_setters_coerce_via_asdistribution():
    rock = make_rock()
    rock.mass = 30.0
    rock.density = stats.Exponential(1.0)
    assert isinstance(rock.mass, stats.Deterministic)
    assert rock.mass.value == 30.0
    assert isinstance(rock.density, stats.Exponential)


def test_num_random_variables_is_two():
    rock = make_rock()
    assert rock.numRandomVariables == 2


# ---------------------------------------------------------------------------
# ppf: numerical mean/variance via a deterministic quantile grid
# ---------------------------------------------------------------------------

def test_ppf_rejects_wrong_shape():
    rock = make_rock()
    with pytest.raises(ValueError, match=r"`q` must have shape \(2, N\)"):
        rock.ppf(np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError, match=r"`q` must have shape \(2, N\)"):
        rock.ppf(np.zeros((3, 10)))


def test_ppf_returns_two_arrays_of_requested_length():
    rock = make_rock()
    n = 50
    q = np.tile(np.linspace(0.01, 0.99, n), (2, 1))
    mass_samples, density_samples = rock.ppf(q)
    assert mass_samples.shape == (n,)
    assert density_samples.shape == (n,)


def test_ppf_matches_analytical_quantiles_at_median():
    rock = make_rock()
    q = np.full((2, 1), 0.5)
    mass_samples, density_samples = rock.ppf(q)
    assert mass_samples[0] == pytest.approx(rock.mass.median())
    assert density_samples[0] == pytest.approx(rock.density.median())


def test_ppf_numerical_mean_and_variance_match_prescribed_distributions():
    rock = make_rock()
    n = 200_000
    # Midpoint quantile grid: mean(ppf(q)) is a quadrature approximation of
    # E[X] = integral_0^1 F^-1(q) dq (and similarly for the second moment),
    # so it should converge to the analytical mean/variance without relying
    # on any randomness.
    grid = (np.arange(n) + 0.5) / n
    q = np.vstack([grid, grid])

    mass_samples, density_samples = rock.ppf(q)

    assert mass_samples.mean() == pytest.approx(rock.mass.mean(), rel=1e-3)
    assert mass_samples.var() == pytest.approx(rock.mass.var(), rel=1e-2)

    assert density_samples.mean() == pytest.approx(rock.density.mean(), rel=1e-3)
    assert density_samples.var() == pytest.approx(rock.density.var(), rel=1e-2)


# ---------------------------------------------------------------------------
# rvs: sample mean/variance vs. the prescribed distributions
# ---------------------------------------------------------------------------

def test_rvs_returns_two_arrays_of_requested_length():
    rock = make_rock()
    mass_samples, density_samples = rock.rvs(100)
    assert mass_samples.shape == (100,)
    assert density_samples.shape == (100,)


def test_rvs_sample_mean_and_variance_match_prescribed_distributions():
    rock = make_rock()
    n = 200_000
    # Rock.rvs() does not expose a random_state parameter; it forwards to
    # scipy's frozen rvs(random_state=None), which reads NumPy's global
    # legacy random state -- seed that directly for a reproducible draw.
    np.random.seed(0)
    mass_samples, density_samples = rock.rvs(n)

    assert mass_samples.mean() == pytest.approx(rock.mass.mean(), rel=5e-2)
    assert mass_samples.var() == pytest.approx(rock.mass.var(), rel=5e-2)

    assert density_samples.mean() == pytest.approx(rock.density.mean(), rel=5e-2)
    assert density_samples.var() == pytest.approx(rock.density.var(), rel=5e-2)


def test_rvs_reproducible_when_global_seed_fixed():
    # Characterizes the lack of a per-call random_state on Rock.rvs: the only
    # way to get reproducible draws is to seed NumPy's global legacy state
    # before each call.
    rock = make_rock()
    np.random.seed(123)
    a_mass, a_density = rock.rvs(50)
    np.random.seed(123)
    b_mass, b_density = rock.rvs(50)
    np.testing.assert_array_equal(a_mass, b_mass)
    np.testing.assert_array_equal(a_density, b_density)
