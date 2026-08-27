"""Tests for pyrockfall.Material: restitution/friction as distributions, rvs/ppf moments.

Material.normalRestitution, .tangentialRestitution and .frictionAngle are
coerced via stats.asDistribution into stats.Distribution objects.
Material.rvs(N) delegates to `<dist>.rvs(N)` for each of the three variables
(no random_state parameter is exposed on Material itself, so reproducibility
relies on seeding NumPy's global legacy random state, exactly like Rock).
Material.ppf(q) expects a sequence of exactly three quantile arrays
`[n, t, f]` and delegates to `<dist>.ppf(...)` for each.

As with test_rock.py, moments are checked two ways:
  * rvs() sample moments (large N) approximate the analytical moments
    (Monte Carlo, loose tolerance), and
  * ppf() moments via a deterministic midpoint quantile grid (quadrature of
    the inverse-CDF) match the analytical moments tightly, with no
    randomness involved.
"""
import numpy as np
import pytest

import pyrockfall as pr
from pyrockfall import stats

NORMAL_ALPHA, NORMAL_BETA = 5.0, 2.0  # Beta(alpha, beta), support [0,1]
TANGENTIAL_ALPHA, TANGENTIAL_BETA = 3.0, 3.0  # Beta(alpha, beta), support [0,1]
FRICTION_MU, FRICTION_SIGMA = 35.0, 5.0  # Normal(mu, sigma), degrees


def make_material():
    return pr.Material(
        name="TestMaterial",
        normalRestitution=stats.Beta(NORMAL_ALPHA, NORMAL_BETA),
        tangentialRestitution=stats.Beta(TANGENTIAL_ALPHA, TANGENTIAL_BETA),
        frictionAngle=stats.Normal(FRICTION_MU, FRICTION_SIGMA),
    )


# ---------------------------------------------------------------------------
# Construction / properties
# ---------------------------------------------------------------------------

def test_properties_are_the_assigned_distributions():
    m = make_material()
    assert isinstance(m.normalRestitution, stats.Beta)
    assert isinstance(m.tangentialRestitution, stats.Beta)
    assert isinstance(m.frictionAngle, stats.Normal)
    assert m.normalRestitution.mean() == pytest.approx(NORMAL_ALPHA / (NORMAL_ALPHA + NORMAL_BETA))
    assert m.frictionAngle.mean() == pytest.approx(FRICTION_MU)


def test_default_properties_are_deterministic_zero():
    m = pr.Material(name="Empty")
    assert isinstance(m.normalRestitution, stats.Deterministic)
    assert isinstance(m.tangentialRestitution, stats.Deterministic)
    assert isinstance(m.frictionAngle, stats.Deterministic)
    assert isinstance(m.roughness, stats.Deterministic)
    assert m.normalRestitution.value == 0.0
    assert m.tangentialRestitution.value == 0.0
    assert m.frictionAngle.value == 0.0
    assert m.roughness.value == 0.0


def test_name_defaults_to_generated_material_name():
    m = pr.Material()
    assert m.name.startswith("Material ")


def test_scalar_properties_are_coerced_to_deterministic():
    m = pr.Material(name="Fixed", normalRestitution=0.4, tangentialRestitution=0.5, frictionAngle=30.0)
    assert isinstance(m.normalRestitution, stats.Deterministic)
    assert m.normalRestitution.value == 0.4
    assert isinstance(m.tangentialRestitution, stats.Deterministic)
    assert m.tangentialRestitution.value == 0.5
    assert isinstance(m.frictionAngle, stats.Deterministic)
    assert m.frictionAngle.value == 30.0


def test_setters_coerce_via_asdistribution():
    m = make_material()
    m.normalRestitution = 0.6
    m.tangentialRestitution = stats.Uniform(0.2, 0.8)
    m.frictionAngle = 40.0
    m.roughness = stats.Exponential(2.0)
    assert isinstance(m.normalRestitution, stats.Deterministic)
    assert m.normalRestitution.value == 0.6
    assert isinstance(m.tangentialRestitution, stats.Uniform)
    assert isinstance(m.frictionAngle, stats.Deterministic)
    assert m.frictionAngle.value == 40.0
    assert isinstance(m.roughness, stats.Exponential)


def test_num_random_variables_is_three():
    m = make_material()
    assert m.numRandomVariables == 3


def test_roughness_is_stored_but_excluded_from_num_random_variables_ppf_and_rvs():
    # Characterization test: `roughness` is a fourth DistributionLike
    # constructor argument with its own getter/setter, but
    # numRandomVariables is hardcoded to 3 and neither ppf() nor rvs()
    # reference roughness at all -- it plays no role in sampling.
    m = pr.Material(name="Rough", roughness=stats.Uniform(0.0, 1.0))
    assert isinstance(m.roughness, stats.Uniform)
    assert m.numRandomVariables == 3
    q = [np.array([0.5]), np.array([0.5]), np.array([0.5])]
    assert m.ppf(q).shape == (3, 1)
    assert m.rvs(4).shape == (3, 4)


# ---------------------------------------------------------------------------
# ppf: numerical mean/variance via a deterministic quantile grid
# ---------------------------------------------------------------------------

def test_ppf_rejects_wrong_length():
    m = make_material()
    with pytest.raises(ValueError, match="`q` must be a sequence of three quantile arrays."):
        m.ppf([np.array([0.5]), np.array([0.5])])
    with pytest.raises(ValueError, match="`q` must be a sequence of three quantile arrays."):
        m.ppf(np.zeros((4, 10)))


def test_ppf_rejects_non_sequence():
    m = make_material()
    with pytest.raises(ValueError, match="`q` must be a sequence of three quantile arrays."):
        m.ppf(0.5)


def test_ppf_returns_array_of_requested_length():
    m = make_material()
    n = 50
    grid = np.linspace(0.01, 0.99, n)
    result = m.ppf([grid, grid, grid])
    assert result.shape == (3, n)


def test_ppf_matches_analytical_quantiles_at_median():
    m = make_material()
    q = [np.array([0.5]), np.array([0.5]), np.array([0.5])]
    result = m.ppf(q)
    assert result[0, 0] == pytest.approx(m.normalRestitution.median())
    assert result[1, 0] == pytest.approx(m.tangentialRestitution.median())
    assert result[2, 0] == pytest.approx(m.frictionAngle.median())


def test_ppf_numerical_mean_and_variance_match_prescribed_distributions():
    m = make_material()
    n = 200_000
    # Midpoint quantile grid: mean(ppf(q)) approximates E[X] = integral_0^1
    # F^-1(q) dq via quadrature, so it converges to the analytical
    # mean/variance without relying on any randomness.
    grid = (np.arange(n) + 0.5) / n
    result = m.ppf([grid, grid, grid])
    normal_samples, tangential_samples, friction_samples = result

    assert normal_samples.mean() == pytest.approx(m.normalRestitution.mean(), rel=1e-3)
    assert normal_samples.var() == pytest.approx(m.normalRestitution.var(), rel=1e-2)

    assert tangential_samples.mean() == pytest.approx(m.tangentialRestitution.mean(), rel=1e-3)
    assert tangential_samples.var() == pytest.approx(m.tangentialRestitution.var(), rel=1e-2)

    assert friction_samples.mean() == pytest.approx(m.frictionAngle.mean(), rel=1e-3)
    assert friction_samples.var() == pytest.approx(m.frictionAngle.var(), rel=1e-2)


# ---------------------------------------------------------------------------
# rvs: sample mean/variance vs. the prescribed distributions
# ---------------------------------------------------------------------------

def test_rvs_rejects_negative_num_samples():
    m = make_material()
    with pytest.raises(ValueError, match="`num_samples` must be non-negative."):
        m.rvs(-1)


def test_rvs_returns_array_of_requested_length():
    m = make_material()
    result = m.rvs(100)
    assert result.shape == (3, 100)


def test_rvs_sample_mean_and_variance_match_prescribed_distributions():
    m = make_material()
    n = 200_000
    # Material.rvs() does not expose a random_state parameter; like Rock, it
    # forwards to rvs(random_state=None), which reads NumPy's global legacy
    # random state -- seed that directly for a reproducible draw.
    np.random.seed(0)
    result = m.rvs(n)
    normal_samples, tangential_samples, friction_samples = result

    assert normal_samples.mean() == pytest.approx(m.normalRestitution.mean(), rel=5e-2)
    assert normal_samples.var() == pytest.approx(m.normalRestitution.var(), rel=5e-2)

    assert tangential_samples.mean() == pytest.approx(m.tangentialRestitution.mean(), rel=5e-2)
    assert tangential_samples.var() == pytest.approx(m.tangentialRestitution.var(), rel=5e-2)

    assert friction_samples.mean() == pytest.approx(m.frictionAngle.mean(), rel=5e-2)
    assert friction_samples.var() == pytest.approx(m.frictionAngle.var(), rel=5e-2)


def test_rvs_reproducible_when_global_seed_fixed():
    m = make_material()
    np.random.seed(123)
    a = m.rvs(50)
    np.random.seed(123)
    b = m.rvs(50)
    np.testing.assert_array_equal(a, b)
