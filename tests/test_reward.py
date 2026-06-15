"""Unit tests for the multi-objective reward."""

from __future__ import annotations

import math

import pytest

from synth_q_mat.rl.reward import (
    RewardWeights,
    compute_reward,
    formation_to_score,
)


def test_weights_normalize_to_one():
    w = RewardWeights(1.0, 1.0, 1.0, 1.0)
    assert math.isclose(w.as_array().sum(), 1.0)
    assert all(math.isclose(x, 0.25) for x in w.as_array())


def test_weights_from_config_roundtrip():
    cfg = {
        "w_topology": 0.5,
        "w_dos_fermi": 0.2,
        "w_formation": 0.2,
        "w_composition": 0.1,
    }
    w = RewardWeights.from_config(cfg)
    assert math.isclose(w.as_array().sum(), 1.0)
    # topology is the largest weight
    assert w.as_array()[0] == max(w.as_array())


def test_zero_weights_rejected():
    with pytest.raises(ValueError):
        RewardWeights(0.0, 0.0, 0.0, 0.0).as_array()


def test_formation_to_score_bounds():
    assert formation_to_score(-0.5) == 1.0  # below hull -> max
    assert formation_to_score(0.0) == 1.0  # on hull -> max
    assert formation_to_score(0.10, cutoff=0.10) == 0.0  # at cutoff -> min
    assert formation_to_score(1.0, cutoff=0.10) == 0.0  # far above -> clipped
    assert 0.0 < formation_to_score(0.05, cutoff=0.10) < 1.0


def test_compute_reward_all_max():
    w = RewardWeights()
    objs = {"topology": 1.0, "dos_fermi": 1.0, "formation": 1.0, "composition": 1.0}
    assert math.isclose(compute_reward(objs, w), 1.0)


def test_compute_reward_weighting():
    # Only topology fires; reward equals topology's normalized weight.
    w = RewardWeights(0.4, 0.3, 0.2, 0.1)
    objs = {"topology": 1.0}
    assert math.isclose(compute_reward(objs, w), 0.4)


def test_compute_reward_missing_objectives_default_zero():
    w = RewardWeights()
    assert compute_reward({}, w) == 0.0


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_compute_reward_rejects_out_of_range(bad):
    w = RewardWeights()
    with pytest.raises(ValueError):
        compute_reward({"topology": bad}, w)
