"""Tests for pyrockfall.Impacts: recording the time history of block impacts
(location, orientation, pre-/post-impact velocities, restitution) and
exposing it as stacked NumPy arrays, in 2D and 3D.

Impacts.addData() records one impact "event" per call across all blocks
(mirroring Trajectories.addData()); simulationDone() then converts every
internal list of per-event snapshots into a single NumPy array.
"""
import numpy as np
import pytest

from pyrockfall._impacts import Impacts

DIMS = [2, 3]


# ---------------------------------------------------------------------------
# Fixture builder: two impact events for a given number of blocks/dims.
# ---------------------------------------------------------------------------

def make_impacts(ndim, n_blocks, n_impacts=2):
    """Builds an Impacts instance with `n_impacts` events for `n_blocks` blocks.

    Values are simple, distinct numbers (event index * 10 + block index) so
    that each stored field can be checked against its exact source value.
    """
    impacts = Impacts()
    angvel_dim = 1 if ndim == 2 else ndim

    raw = {
        "time": [], "position": [], "normal": [], "element_id": [],
        "normal_impact_velocity": [], "tangential_impact_velocity": [], "angular_impact_velocity": [],
        "normal_post_velocity": [], "tangential_post_velocity": [], "angular_post_velocity": [],
        "normal_restitution": [], "tangential_restitution": [],
    }

    for e in range(n_impacts):
        base = e * 10.0
        time = base + np.arange(n_blocks, dtype=float)
        position = base + np.arange(ndim * n_blocks, dtype=float).reshape(ndim, n_blocks)
        normal = base + np.arange(ndim * n_blocks, dtype=float).reshape(ndim, n_blocks) + 100.0
        element_id = (base + np.arange(n_blocks)).astype(int)
        normal_impact_velocity = base + np.arange(n_blocks, dtype=float) + 1.0
        tangential_impact_velocity = base + np.arange(n_blocks, dtype=float) + 2.0
        angular_impact_velocity = base + np.arange(angvel_dim * n_blocks, dtype=float).reshape(angvel_dim, n_blocks)
        normal_post_velocity = base + np.arange(n_blocks, dtype=float) + 3.0
        tangential_post_velocity = base + np.arange(n_blocks, dtype=float) + 4.0
        angular_post_velocity = base + np.arange(angvel_dim * n_blocks, dtype=float).reshape(angvel_dim, n_blocks) + 1.0
        normal_restitution = np.full(n_blocks, 0.5 + e * 0.1)
        tangential_restitution = np.full(n_blocks, 0.4 + e * 0.1)

        impacts.addData(
            time, position, normal, element_id,
            normal_impact_velocity, tangential_impact_velocity, angular_impact_velocity,
            normal_post_velocity, tangential_post_velocity, angular_post_velocity,
            normal_restitution, tangential_restitution,
        )

        raw["time"].append(time)
        raw["position"].append(position)
        raw["normal"].append(normal)
        raw["element_id"].append(element_id)
        raw["normal_impact_velocity"].append(normal_impact_velocity)
        raw["tangential_impact_velocity"].append(tangential_impact_velocity)
        raw["angular_impact_velocity"].append(angular_impact_velocity)
        raw["normal_post_velocity"].append(normal_post_velocity)
        raw["tangential_post_velocity"].append(tangential_post_velocity)
        raw["angular_post_velocity"].append(angular_post_velocity)
        raw["normal_restitution"].append(normal_restitution)
        raw["tangential_restitution"].append(tangential_restitution)

    impacts.simulationDone()
    return impacts, raw


# ---------------------------------------------------------------------------
# addData / simulationDone: lists become arrays with the expected shapes
# and values.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ndim", DIMS)
def test_history_properties_have_expected_shapes(ndim):
    n_blocks, n_impacts = 3, 2
    impacts, raw = make_impacts(ndim, n_blocks, n_impacts)

    angvel_dim = 1 if ndim == 2 else ndim
    assert impacts.timeHistory.shape == (n_impacts, n_blocks)
    assert impacts.positionHistory.shape == (n_impacts, ndim, n_blocks)
    assert impacts.normalHistory.shape == (n_impacts, ndim, n_blocks)
    assert impacts.elementIdHistory.shape == (n_impacts, n_blocks)
    assert impacts.normalImpactVelocityHistory.shape == (n_impacts, n_blocks)
    assert impacts.tangentialImpactVelocityHistory.shape == (n_impacts, n_blocks)
    assert impacts.angularImpactVelocityHistory.shape == (n_impacts, angvel_dim, n_blocks)
    assert impacts.normalPostVelocityHistory.shape == (n_impacts, n_blocks)
    assert impacts.tangentialPostVelocityHistory.shape == (n_impacts, n_blocks)
    assert impacts.angularPostVelocityHistory.shape == (n_impacts, angvel_dim, n_blocks)
    assert impacts.normalRestitutionHistory.shape == (n_impacts, n_blocks)
    assert impacts.tangentialRestitutionHistory.shape == (n_impacts, n_blocks)


@pytest.mark.parametrize("ndim", DIMS)
def test_history_values_match_what_was_added(ndim):
    n_blocks, n_impacts = 2, 3
    impacts, raw = make_impacts(ndim, n_blocks, n_impacts)

    np.testing.assert_allclose(impacts.timeHistory, np.stack(raw["time"]))
    np.testing.assert_allclose(impacts.positionHistory, np.stack(raw["position"]))
    np.testing.assert_allclose(impacts.normalHistory, np.stack(raw["normal"]))
    np.testing.assert_allclose(impacts.elementIdHistory, np.stack(raw["element_id"]))
    np.testing.assert_allclose(impacts.normalImpactVelocityHistory, np.stack(raw["normal_impact_velocity"]))
    np.testing.assert_allclose(impacts.tangentialImpactVelocityHistory, np.stack(raw["tangential_impact_velocity"]))
    np.testing.assert_allclose(impacts.angularImpactVelocityHistory, np.stack(raw["angular_impact_velocity"]))
    np.testing.assert_allclose(impacts.normalPostVelocityHistory, np.stack(raw["normal_post_velocity"]))
    np.testing.assert_allclose(impacts.tangentialPostVelocityHistory, np.stack(raw["tangential_post_velocity"]))
    np.testing.assert_allclose(impacts.angularPostVelocityHistory, np.stack(raw["angular_post_velocity"]))
    np.testing.assert_allclose(impacts.normalRestitutionHistory, np.stack(raw["normal_restitution"]))
    np.testing.assert_allclose(impacts.tangentialRestitutionHistory, np.stack(raw["tangential_restitution"]))


def test_add_data_accumulates_before_simulation_done():
    impacts = Impacts()
    impacts.addData(0.0, [1.0, 2.0], [0.0, 1.0], 5,
                     1.0, 0.5, 0.2, 0.8, 0.3, 0.1, 0.9, 0.8)
    impacts.addData(1.0, [3.0, 4.0], [0.0, 1.0], 6,
                     1.1, 0.6, 0.3, 0.9, 0.4, 0.2, 0.85, 0.75)
    assert len(impacts._time_history) == 2
    impacts.simulationDone()
    assert isinstance(impacts._time_history, np.ndarray)
    assert impacts.timeHistory.shape == (2,)


# ---------------------------------------------------------------------------
# numberOfBlocks / numberOfImpacts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ndim", DIMS)
def test_number_of_blocks_and_number_of_impacts(ndim):
    impacts, raw = make_impacts(ndim, n_blocks=4, n_impacts=3)
    assert impacts.numberOfBlocks == 4
    assert impacts.numberOfImpacts == 3


# ---------------------------------------------------------------------------
# impactPoints / impactNormalVelocities / impactTangentialVelocities /
# impactAngularVelocities: moved from Trajectories, transposed history.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ndim", DIMS)
def test_impact_points_is_transposed_position_history(ndim):
    impacts, raw = make_impacts(ndim, n_blocks=3, n_impacts=2)
    np.testing.assert_allclose(impacts.impactPoints, np.transpose(impacts.positionHistory, (1, 0, 2)))
    assert impacts.impactPoints.shape == (ndim, 2, 3)


@pytest.mark.parametrize("ndim", DIMS)
def test_impact_normal_and_tangential_velocities_are_transposed_history(ndim):
    impacts, raw = make_impacts(ndim, n_blocks=3, n_impacts=2)
    np.testing.assert_allclose(impacts.impactNormalVelocities, np.transpose(impacts.normalImpactVelocityHistory))
    np.testing.assert_allclose(
        impacts.impactTangentialVelocities, np.transpose(impacts.tangentialImpactVelocityHistory)
    )
    assert impacts.impactNormalVelocities.shape == (3, 2)
    assert impacts.impactTangentialVelocities.shape == (3, 2)


@pytest.mark.parametrize("ndim", DIMS)
def test_impact_angular_velocities_is_transposed_history(ndim):
    impacts, raw = make_impacts(ndim, n_blocks=3, n_impacts=2)
    angvel_dim = 1 if ndim == 2 else ndim
    np.testing.assert_allclose(
        impacts.impactAngularVelocities, np.transpose(impacts.angularImpactVelocityHistory, (1, 0, 2))
    )
    assert impacts.impactAngularVelocities.shape == (angvel_dim, 2, 3)
