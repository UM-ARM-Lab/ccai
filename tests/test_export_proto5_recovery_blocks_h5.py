import math
from argparse import Namespace

import h5py
import numpy as np
import pytest
import torch

from scripts import export_proto5_recovery_blocks_h5 as exporter


def _arrays(states, *, mode="thumb_middle"):
    action_count = len(states) - 1
    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.zeros((action_count, 12), dtype=np.float32),
        "contact_plan": np.ones((action_count, 3), dtype=np.float32),
        "contact_state": np.zeros((len(states), 3), dtype=np.float32),
        "contact_wrenches": np.zeros((len(states), 3, 6), dtype=np.float32),
        "contact_forces": np.zeros((len(states), 3, 3), dtype=np.float32),
        "contact_points": np.zeros((len(states), 3, 3), dtype=np.float32),
        "contact_modes": np.asarray([mode] * action_count, dtype=object),
        "episode_num_steps": np.arange(action_count, dtype=np.int64),
        "stage_index": np.arange(10, 10 + action_count, dtype=np.int64),
        "recover": np.ones((action_count,), dtype=np.bool_),
        "record_likelihoods": np.full((action_count,), np.nan, dtype=np.float32),
    }


def _block(states, *, terminal_reason="switch_to_turn", initial=-10.0, final=-9.0, block_idx=0):
    return {
        "arrays": _arrays(states),
        "action_length": len(states) - 1,
        "state_length": len(states),
        "source_experiment": "exp",
        "source_trial_dir": 1,
        "trial_index": 0,
        "trial_initial_state": np.asarray(states[0], dtype=np.float32),
        "screwdriver_friction": 2.5,
        "yaw_joint_friction": 0.03,
        "block_index_in_trial": block_idx,
        "start_contact_index": block_idx,
        "end_contact_index": block_idx,
        "start_stage_index": 10 + block_idx,
        "end_stage_index": 10 + block_idx,
        "start_episode_num_steps": block_idx,
        "end_episode_num_steps": block_idx + len(states) - 2,
        "initial_likelihood": initial,
        "final_likelihood": final,
        "likelihood_delta": final - initial,
        "improved_likelihood": final > initial,
        "terminal_reason": terminal_reason,
    }


class _YawReturningSampler:
    def __init__(self):
        self.calls = []

    def check_id_batch(self, states, n, likelihood_only=False):
        assert n == 8
        assert likelihood_only is True
        yaws = states[:, -1].detach().cpu()
        self.calls.extend(float(yaw) for yaw in yaws)
        return yaws.to(device=states.device, dtype=states.dtype)


class _MeanGroupedSampler:
    def __init__(self):
        self.batches = []

    def check_id_batch(self, states, n, likelihood_only=False):
        assert likelihood_only is True
        self.batches.append(states.detach().cpu().clone())
        base = torch.arange(states.shape[0], device=states.device, dtype=states.dtype) * 10.0
        samples = base.unsqueeze(1) + torch.arange(n, device=states.device, dtype=states.dtype)
        return samples.mean(dim=1)


def test_recompute_block_likelihoods_updates_yaw_before_switch_to_turn_terminal(monkeypatch):
    states_1 = [
        np.zeros(15, dtype=np.float32),
        np.array([0.0] * 14 + [-(math.pi / 2.0) - 0.1], dtype=np.float32),
        np.array([0.0] * 14 + [-(math.pi / 2.0) - 0.2], dtype=np.float32),
    ]
    states_2 = [
        np.array([0.0] * 14 + [-(math.pi / 2.0) - 0.2], dtype=np.float32),
        np.array([0.0] * 14 + [-(math.pi / 2.0) - 0.3], dtype=np.float32),
        np.array([0.0] * 14 + [-(math.pi / 2.0) - 0.4], dtype=np.float32),
    ]
    blocks = [
        _block(states_1, block_idx=0),
        _block(states_2, terminal_reason="episode_end", block_idx=1),
    ]
    exporter.attach_original_likelihoods(blocks)
    blocks[0]["original_likelihood"][1] = np.float32(-8.0)
    blocks[0]["original_likelihood"][2] = np.nan
    blocks[0]["likelihood_originally_labeled_mask"] = np.isfinite(blocks[0]["original_likelihood"])
    sampler = _YawReturningSampler()
    monkeypatch.setattr(
        exporter,
        "load_labeling_sampler",
        lambda args: (
            {"likelihood_num_samples": 8, "model_path": "model.pt", "config_path": "cfg.yaml", "sine_cosine": True, "T_orig": 6, "type": "diffusion", "compile_models": False},
            {"device": "cpu"},
            sampler,
            torch,
        ),
    )

    summary = {}
    exporter.recompute_block_likelihoods(
        blocks,
        Namespace(skip_likelihood_relabel=False, likelihood_batch_size=64),
        summary,
    )

    assert sampler.calls == pytest.approx([-0.2, -0.3], abs=1e-6)
    assert blocks[0]["likelihood"].tolist() == pytest.approx([-10.0, -8.0, -0.2], abs=1e-6)
    assert blocks[1]["likelihood"].tolist() == pytest.approx([-10.0, -0.3, -9.0], abs=1e-6)
    np.testing.assert_array_equal(blocks[0]["likelihood_recomputed_mask"], [False, False, True])
    np.testing.assert_array_equal(blocks[1]["likelihood_recomputed_mask"], [False, True, False])
    assert summary["likelihood_relabeling"]["original_likelihood_states_preserved"] == 4
    assert summary["likelihood_relabeling"]["missing_likelihood_states_filled"] == 2
    assert summary["likelihood_relabeling"]["missing_likelihood_states_unfilled"] == 0
    assert summary["likelihood_relabeling"]["likelihood_batches"] == 1


def test_recompute_block_likelihoods_preserves_originals_and_batches_only_missing(monkeypatch):
    states = [
        np.zeros(15, dtype=np.float32),
        np.ones(15, dtype=np.float32),
        np.ones(15, dtype=np.float32) * 2.0,
        np.ones(15, dtype=np.float32) * 3.0,
        np.ones(15, dtype=np.float32) * 4.0,
    ]
    block = _block(states, initial=-7.0, final=-1.0)
    block["arrays"]["record_likelihoods"] = np.asarray([-6.0, np.nan, np.nan, np.nan], dtype=np.float32)
    blocks = [block]
    exporter.attach_original_likelihoods(blocks)
    sampler = _MeanGroupedSampler()
    monkeypatch.setattr(
        exporter,
        "load_labeling_sampler",
        lambda args: (
            {"likelihood_num_samples": 4, "model_path": "model.pt", "config_path": "cfg.yaml", "sine_cosine": True, "T_orig": 6, "type": "diffusion", "compile_models": False},
            {"device": "cpu"},
            sampler,
            torch,
        ),
    )

    summary = {}
    exporter.recompute_block_likelihoods(
        blocks,
        Namespace(skip_likelihood_relabel=False, likelihood_batch_size=64),
        summary,
    )

    assert len(sampler.batches) == 1
    assert sampler.batches[0].shape[0] == 2
    np.testing.assert_allclose(block["likelihood"], [-7.0, -6.0, 1.5, 11.5, -1.0], atol=1e-6)
    np.testing.assert_array_equal(block["likelihood_originally_labeled_mask"], [True, True, False, False, True])
    np.testing.assert_array_equal(block["likelihood_recomputed_mask"], [False, False, True, True, False])
    assert summary["likelihood_relabeling"]["original_likelihood_states_preserved"] == 3
    assert summary["likelihood_relabeling"]["missing_likelihood_states_filled"] == 2
    assert summary["likelihood_relabeling"]["likelihood_batches"] == 1


def test_write_blocks_h5_uses_block_rows_masks_and_action_aligned_metadata(tmp_path):
    states = [
        np.arange(15, dtype=np.float32),
        np.arange(15, dtype=np.float32) + 1,
        np.arange(15, dtype=np.float32) + 2,
    ]
    short_states = [
        np.arange(15, dtype=np.float32) + 10,
        np.arange(15, dtype=np.float32) + 11,
    ]
    block = _block(states)
    short_block = _block(short_states, block_idx=1)
    blocks = [block, short_block]
    exporter.attach_original_likelihoods(blocks)
    block["likelihood"] = np.asarray([-1.0, -2.0, -3.0], dtype=np.float32)
    block["likelihood_recomputed_mask"] = np.asarray([False, True, False], dtype=np.bool_)
    short_block["likelihood"] = np.asarray([-4.0, -5.0], dtype=np.float32)
    short_block["likelihood_recomputed_mask"] = np.asarray([False, False], dtype=np.bool_)
    exporter.refresh_endpoint_likelihood_fields(blocks, {})

    output = tmp_path / "blocks.h5"
    exporter.write_blocks_h5(blocks, output, {"sources": []})

    with h5py.File(output, "r") as h5:
        assert h5["q"].shape == (2, 3, 3, 4)
        assert h5["observation"].shape == (2, 3, 3)
        assert h5["action"].shape == (2, 2, 3, 4)
        assert h5["likelihood"].shape == h5["q"].shape[:2]
        assert h5["original_likelihood"].shape == h5["q"].shape[:2]
        assert h5["trajectory_lengths"][:].tolist() == [3, 2]
        assert h5["action_lengths"][:].tolist() == [2, 1]
        np.testing.assert_array_equal(h5["valid_state_mask"][:], [[True, True, True], [True, True, False]])
        np.testing.assert_array_equal(h5["valid_action_mask"][:], [[True, True], [True, False]])
        assert np.isnan(h5["likelihood"][1, 2])
        np.testing.assert_array_equal(
            h5["likelihood_recomputed_mask"][:],
            h5["valid_state_mask"][:] & ~h5["likelihood_originally_labeled_mask"][:],
        )
        assert h5["episode_num_steps"].shape == (2, 2)
        assert h5["stage_index"].shape == (2, 2)
        assert h5["recover"].shape == (2, 2)
        assert h5["contact_mode"].shape == (2, 2)
        assert h5.attrs["online_likelihood_threshold"] == -90.0
