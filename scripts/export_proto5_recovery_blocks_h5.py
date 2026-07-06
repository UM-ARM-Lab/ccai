#!/usr/bin/env python3
"""Export Proto5 recovery blocks as one H5 row per recovery attempt."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


CCAI_PATH = Path(__file__).resolve().parents[1]
DEFAULT_LABEL_CONFIG = CCAI_PATH / "examples/config/proto5/proto_screwdriver_csvto_TODR_recovery_data_gen.yaml"
DEFAULT_LABEL_MODEL = (
    CCAI_PATH
    / "data/training/allegro_screwdriver/proto5_blue_screwdriver_task_fixed_replay_diffusion/"
    "allegro_screwdriver_diffusion_best.pt"
)
ONLINE_LIKELIHOOD_THRESHOLD = -90.0


DEFAULT_EXPERIMENT_DIRS = (
    "data/experiments/proto_blue_screwdriver_D_TOUR_recovery_data_gen_fixed_replay_90",
    "data/experiments/proto_blue_screwdriver_D_TOUR_recovery_data_gen_fixed_replay_90_1",
    "data/experiments/proto_blue_screwdriver_D_TOUR_recovery_data_gen_fixed_replay_90_no_belief_reset",
    "data/experiments/proto_blue_screwdriver_D_TOUR_recovery_data_gen_fixed_replay_90_no_belief_reset_1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment_dirs",
        nargs="*",
        type=Path,
        default=[Path(path) for path in DEFAULT_EXPERIMENT_DIRS],
        help="Experiment folders containing csvgd/trial_*/traj_data.p.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trials-per-folder",
        type=int,
        default=None,
        help="Optional smoke-test cap: use only the first N trial directories per folder.",
    )
    parser.add_argument(
        "--allow-empty-blocks",
        action="store_true",
        help="Include recovery stages that have no executed recovery action rows.",
    )
    parser.add_argument(
        "--label-config",
        type=Path,
        default=DEFAULT_LABEL_CONFIG,
        help="Recovery YAML used to construct the task diffusion likelihood sampler.",
    )
    parser.add_argument(
        "--label-model",
        type=Path,
        default=DEFAULT_LABEL_MODEL,
        help="Task diffusion checkpoint used to recompute dense state-aligned likelihood labels.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the labeling device. Defaults to controllers.csvgd.device from --label-config.",
    )
    parser.add_argument(
        "--skip-likelihood-relabel",
        action="store_true",
        help="Do not load the sampler; fill dense likelihood from original logged values only.",
    )
    parser.add_argument(
        "--likelihood-batch-size",
        type=int,
        default=64,
        help="Number of missing valid states to label per sampler batch.",
    )
    return parser.parse_args()


def to_float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (list, tuple)):
        if not value:
            return float("nan")
        value = value[-1]
    if value is None:
        return float("nan")
    arr = np.asarray(value)
    if arr.size == 0:
        return float("nan")
    value = arr.reshape(-1)[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def stage_value(series: list[Any], stage_index: int) -> float:
    if stage_index < 0 or stage_index >= len(series):
        return float("nan")
    return to_float_or_nan(series[stage_index])


def previous_likelihood(pre_action_likelihoods: list[Any], final_likelihoods: list[Any], start_contact_idx: int) -> float:
    for contact_idx in range(start_contact_idx - 1, -1, -1):
        value = stage_value(pre_action_likelihoods, contact_idx)
        if not math.isnan(value):
            return value
        value = stage_value(final_likelihoods, contact_idx)
        if not math.isnan(value):
            return value
    return float("nan")


def record_stage_index(record: dict[str, Any]) -> int:
    return int(record.get("stage_index", -1))


def record_step(record: dict[str, Any]) -> int:
    return int(record.get("episode_num_steps", -1))


def recovery_records_for_block(records: list[dict[str, Any]], stage_indices: set[int]) -> list[dict[str, Any]]:
    block_records = [
        record
        for record in records
        if bool(record.get("recover", False)) and record_stage_index(record) in stage_indices
    ]
    return sorted(block_records, key=lambda record: (record_step(record), record_stage_index(record)))


def recovery_sequence_blocks(
    executed_contacts: list[str],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records_by_stage: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_stage[record_stage_index(record)].append(record)

    blocks: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None

    def close_active(reason: str) -> None:
        nonlocal active
        if active is not None:
            active["terminal_reason"] = reason
            blocks.append(active)
            active = None

    for stage_index, _contact in enumerate(executed_contacts, start=1):
        stage_records = records_by_stage.get(stage_index, [])
        has_recovery = any(bool(record.get("recover", False)) for record in stage_records)
        has_non_recovery = any(not bool(record.get("recover", False)) for record in stage_records)

        if has_recovery:
            if active is None:
                active = {
                    "start_contact_index": stage_index - 1,
                    "end_contact_index": stage_index - 1,
                    "recovery_stage_indices": [],
                }
            active["end_contact_index"] = stage_index - 1
            active["recovery_stage_indices"].append(stage_index)
            continue

        if active is not None and has_non_recovery:
            close_active("return_to_id")

    close_active("episode_end")
    return blocks


def stack_block_records(block_records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    first = block_records[0]
    states = [np.asarray(first["states"], dtype=np.float32)[0]]
    contact_state = [np.asarray(first["contact_state"], dtype=np.float32)[0]]
    contact_wrenches = [np.asarray(first["contact_wrenches"], dtype=np.float32)[0]]
    contact_forces = [np.asarray(first["contact_forces"], dtype=np.float32)[0]]
    contact_points = [np.asarray(first["contact_points"], dtype=np.float32)[0]]
    actions = []
    contact_plan = []
    contact_modes = []
    episode_num_steps = []
    stage_indices = []
    recover = []
    record_likelihoods = []
    terminal_state_indices = []
    previous_stage_index = None
    for record in block_records:
        stage_index = int(record.get("stage_index", -1))
        if previous_stage_index is not None and stage_index != previous_stage_index:
            terminal_state_indices.append(len(states) - 1)
        previous_stage_index = stage_index
        states_arr = np.asarray(record["states"], dtype=np.float32)
        actions_arr = np.asarray(record["actions"], dtype=np.float32)
        contact_state_arr = np.asarray(record["contact_state"], dtype=np.float32)
        contact_wrenches_arr = np.asarray(record["contact_wrenches"], dtype=np.float32)
        contact_forces_arr = np.asarray(record["contact_forces"], dtype=np.float32)
        contact_points_arr = np.asarray(record["contact_points"], dtype=np.float32)
        actions.append(actions_arr.reshape(-1, 12)[0])
        contact_plan.append(np.asarray(record["contact_plan"], dtype=np.float32).reshape(-1, 3)[0])
        states.append(states_arr[1])
        contact_state.append(contact_state_arr[1])
        contact_wrenches.append(contact_wrenches_arr[1])
        contact_forces.append(contact_forces_arr[1])
        contact_points.append(contact_points_arr[1])
        contact_modes.append(str(record.get("contact_mode", "")))
        episode_num_steps.append(int(record.get("episode_num_steps", -1)))
        stage_indices.append(int(record.get("stage_index", -1)))
        recover.append(bool(record.get("recover", False)))
        record_likelihoods.append(to_float_or_nan(record.get("likelihood")))
    return {
        "states": np.stack(states, axis=0).astype(np.float32),
        "actions": np.stack(actions, axis=0).astype(np.float32),
        "contact_plan": np.stack(contact_plan, axis=0).astype(np.float32),
        "contact_state": np.stack(contact_state, axis=0).astype(np.float32),
        "contact_wrenches": np.stack(contact_wrenches, axis=0).astype(np.float32),
        "contact_forces": np.stack(contact_forces, axis=0).astype(np.float32),
        "contact_points": np.stack(contact_points, axis=0).astype(np.float32),
        "contact_modes": np.asarray(contact_modes, dtype=object),
        "episode_num_steps": np.asarray(episode_num_steps, dtype=np.int64),
        "stage_index": np.asarray(stage_indices, dtype=np.int64),
        "recover": np.asarray(recover, dtype=np.bool_),
        "record_likelihoods": np.asarray(record_likelihoods, dtype=np.float32),
        "recovery_terminal_state_indices": np.asarray([*terminal_state_indices, len(states) - 1], dtype=np.int64),
    }


def read_blocks_from_trial(
    traj_data_path: Path,
    *,
    source_experiment: str,
    allow_empty_blocks: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    with open(traj_data_path, "rb") as handle:
        traj_data = pickle.load(handle)
    executed_contacts = [str(contact) for contact in traj_data.get("executed_contacts", [])]
    records = list(traj_data.get("hri_diffpf_records", []))
    pre_action_likelihoods = list(traj_data.get("pre_action_likelihoods", []))
    final_likelihoods = list(traj_data.get("final_likelihoods", []))
    trial_dir_index = int(traj_data_path.parent.name.split("_")[-1])
    trial_initial_state = None
    if records:
        first_states = np.asarray(records[0].get("states", []), dtype=np.float32)
        if first_states.size:
            trial_initial_state = first_states.reshape(-1, first_states.shape[-1])[0, :15].copy()
    blocks: list[dict[str, Any]] = []
    stats = defaultdict(int)
    for block_idx, sequence_block in enumerate(recovery_sequence_blocks(executed_contacts, records)):
        start_contact_idx = int(sequence_block["start_contact_index"])
        end_contact_idx = int(sequence_block["end_contact_index"])
        stage_indices = set(int(stage_index) for stage_index in sequence_block["recovery_stage_indices"])
        block_records = recovery_records_for_block(records, stage_indices)
        if not block_records:
            stats["empty_blocks"] += 1
            if not allow_empty_blocks:
                continue
        if block_records:
            arrays = stack_block_records(block_records)
            action_length = int(arrays["actions"].shape[0])
            trial_index = int(block_records[0].get("trial_index", trial_dir_index))
            start_step = int(block_records[0].get("episode_num_steps", -1))
            end_step = int(block_records[-1].get("episode_num_steps", -1))
            screwdriver_friction = float(block_records[0].get("screwdriver_friction", np.nan))
            yaw_joint_friction = float(block_records[0].get("yaw_joint_friction", np.nan))
        else:
            arrays = None
            action_length = 0
            trial_index = trial_dir_index
            start_step = -1
            end_step = -1
            screwdriver_friction = float("nan")
            yaw_joint_friction = float("nan")
        start_likelihood = previous_likelihood(pre_action_likelihoods, final_likelihoods, start_contact_idx)
        final_likelihood = stage_value(final_likelihoods, end_contact_idx)
        terminal_likelihoods = []
        for stage_index in sorted(stage_indices):
            terminal_likelihoods.append(stage_value(final_likelihoods, stage_index - 1))
        blocks.append(
            {
                "arrays": arrays,
                "action_length": action_length,
                "state_length": action_length + 1,
                "source_experiment": source_experiment,
                "source_trial_dir": trial_dir_index,
                "trial_index": trial_index,
                "trial_initial_state": trial_initial_state,
                "screwdriver_friction": screwdriver_friction,
                "yaw_joint_friction": yaw_joint_friction,
                "block_index_in_trial": block_idx,
                "start_contact_index": start_contact_idx,
                "end_contact_index": end_contact_idx,
                "start_stage_index": start_contact_idx + 1,
                "end_stage_index": end_contact_idx + 1,
                "recovery_stage_indices": sorted(stage_indices),
                "start_episode_num_steps": start_step,
                "end_episode_num_steps": end_step,
                "initial_likelihood": start_likelihood,
                "final_likelihood": final_likelihood,
                "recovery_terminal_likelihoods": terminal_likelihoods,
                "likelihood_delta": final_likelihood - start_likelihood,
                "improved_likelihood": (
                    (not math.isnan(start_likelihood))
                    and (not math.isnan(final_likelihood))
                    and final_likelihood > start_likelihood
                ),
                "terminal_reason": str(sequence_block["terminal_reason"]),
            }
        )
    return blocks, dict(stats)


def collect_blocks(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"sources": []}
    for experiment_dir in args.experiment_dirs:
        trial_paths = sorted(
            (experiment_dir / "csvgd").glob("trial_*/traj_data.p"),
            key=lambda path: int(path.parent.name.split("_")[-1]),
        )
        if args.trials_per_folder is not None:
            trial_paths = trial_paths[: int(args.trials_per_folder)]
        source_blocks: list[dict[str, Any]] = []
        source_empty = 0
        for traj_data_path in trial_paths:
            trial_blocks, stats = read_blocks_from_trial(
                traj_data_path,
                source_experiment=experiment_dir.name,
                allow_empty_blocks=bool(args.allow_empty_blocks),
            )
            source_empty += int(stats.get("empty_blocks", 0))
            source_blocks.extend(trial_blocks)
        blocks.extend(source_blocks)
        summary["sources"].append(
            {
                "experiment_dir": str(experiment_dir),
                "trial_files": len(trial_paths),
                "recovery_blocks": len(source_blocks),
                "empty_blocks": source_empty,
            }
        )
    return blocks, summary


def fill_padded(dest: np.ndarray, index: int, value: np.ndarray) -> None:
    slices = tuple(slice(0, dim) for dim in value.shape)
    dest[(index, *slices)] = value


def relative_to_ccai(path: Path) -> str:
    path = path.expanduser()
    if not path.is_absolute():
        path = (CCAI_PATH / path).resolve()
    try:
        return str(path.resolve().relative_to(CCAI_PATH))
    except ValueError as exc:
        raise ValueError(f"Label model must be inside {CCAI_PATH}: {path}") from exc


def load_labeling_sampler(args: argparse.Namespace):
    import yaml

    if str(CCAI_PATH) not in sys.path:
        sys.path.insert(0, str(CCAI_PATH))
    import torch

    from ccai.models.management.model_manager import ModelManager

    config_path = args.label_config
    if not config_path.is_absolute():
        config_path = CCAI_PATH / config_path
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["config_path"] = str(config_path)
    config["model_path"] = relative_to_ccai(args.label_model)
    config["task_model_path"] = None
    config["likelihood_num_samples"] = 8
    config["sine_cosine"] = True
    config["T_orig"] = 6
    config["type"] = "diffusion"
    config["compile_models"] = False
    config["obj_dof"] = 3
    config.setdefault("recovery_controller", "csvgd")

    params = config.copy()
    params.pop("controllers", None)
    params.update(config.get("controllers", {}).get("csvgd", {}))
    if args.device is not None:
        params["device"] = args.device
    params.setdefault("device", config.get("sim_device", "cuda:0"))
    params["task_model_path"] = None
    params["recovery_controller"] = config["recovery_controller"]

    manager = ModelManager(config, params, CCAI_PATH)
    _, trajectory_sampler_orig, _ = manager.load_trajectory_samplers(obj_dof=3)
    if trajectory_sampler_orig is None:
        raise RuntimeError("ModelManager did not return a task trajectory sampler for labeling.")
    trajectory_sampler_orig.eval()
    return config, params, trajectory_sampler_orig, torch


def recompute_block_likelihoods(
    blocks: list[dict[str, Any]],
    args: argparse.Namespace,
    summary: dict[str, Any],
) -> None:
    original_likelihood_states_preserved = 0
    for block in blocks:
        state_length = int(block["state_length"])
        block["likelihood"] = np.full((state_length,), np.nan, dtype=np.float32)
        block["likelihood_recomputed_mask"] = np.zeros((state_length,), dtype=np.bool_)
        original = block.get("original_likelihood")
        if original is not None:
            finite = np.isfinite(original)
            block["likelihood"][finite] = original[finite]
            original_likelihood_states_preserved += int(np.count_nonzero(finite))

    if args.skip_likelihood_relabel:
        missing_valid_states = sum(
            int(block["state_length"]) - int(np.count_nonzero(np.isfinite(block.get("original_likelihood", []))))
            for block in blocks
            if block["arrays"] is not None
        )
        summary["likelihood_relabeling"] = {
            "enabled": False,
            "original_likelihood_states_preserved": original_likelihood_states_preserved,
            "missing_likelihood_states_filled": 0,
            "missing_likelihood_states_unfilled": missing_valid_states,
            "likelihood_batches": 0,
            "likelihood_batch_size": int(args.likelihood_batch_size),
        }
        return

    config, params, trajectory_sampler_orig, torch = load_labeling_sampler(args)
    from ccai.utils.screwdriver_yaw_wrap import (
        reset_screwdriver_yaw_wrap,
        update_screwdriver_yaw_wrap_after_recovery,
        wrap_screwdriver_task_state_yaw,
    )

    blocks_by_trial: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        if block["arrays"] is not None:
            blocks_by_trial[(str(block["source_experiment"]), int(block["source_trial_dir"]))].append(block)

    if int(args.likelihood_batch_size) <= 0:
        raise ValueError("--likelihood-batch-size must be positive.")

    pending_states: list[Any] = []
    pending_targets: list[tuple[dict[str, Any], int]] = []
    missing_likelihood_states_total = 0
    missing_likelihood_states_filled = 0
    likelihood_batches = 0

    def flush_pending() -> None:
        nonlocal missing_likelihood_states_filled, likelihood_batches
        if not pending_states:
            return

        states_t = torch.stack(pending_states, dim=0)
        if hasattr(trajectory_sampler_orig, "check_id_batch"):
            likelihoods = trajectory_sampler_orig.check_id_batch(
                states_t,
                int(config["likelihood_num_samples"]),
                likelihood_only=True,
            )
        else:
            likelihoods = [
                trajectory_sampler_orig.check_id(
                    state,
                    int(config["likelihood_num_samples"]),
                    likelihood_only=True,
                )
                for state in states_t
            ]
            likelihoods = torch.as_tensor(likelihoods, device=states_t.device, dtype=states_t.dtype)

        likelihood_batches += 1
        likelihoods = likelihoods.detach().cpu().reshape(-1).numpy()
        if len(likelihoods) != len(pending_targets):
            raise RuntimeError(
                f"Sampler returned {len(likelihoods)} likelihoods for {len(pending_targets)} queued states."
            )
        for likelihood, (block, state_idx) in zip(likelihoods, pending_targets):
            if np.isfinite(likelihood):
                block["likelihood"][state_idx] = np.float32(likelihood)
                block["likelihood_recomputed_mask"][state_idx] = True
                missing_likelihood_states_filled += 1
        pending_states.clear()
        pending_targets.clear()

    for trial_blocks in blocks_by_trial.values():
        trial_blocks.sort(key=lambda block: int(block["block_index_in_trial"]))
        initial_state = next(
            (block.get("trial_initial_state") for block in trial_blocks if block.get("trial_initial_state") is not None),
            None,
        )
        if initial_state is None:
            initial_state = trial_blocks[0]["arrays"]["states"][0]
        wrap_params = dict(params)
        reset_screwdriver_yaw_wrap(wrap_params, torch.as_tensor(initial_state, dtype=torch.float32))
        for block in trial_blocks:
            states = block["arrays"]["states"]
            original_mask = block["likelihood_originally_labeled_mask"]
            terminal_state_indices = set(
                int(idx) for idx in block["arrays"].get("recovery_terminal_state_indices", [])
            )
            if not terminal_state_indices and block["terminal_reason"] == "switch_to_turn":
                terminal_state_indices.add(len(states) - 1)
            for state_idx, state in enumerate(states):
                state_t = torch.as_tensor(state, device=params["device"], dtype=torch.float32)
                if state_idx in terminal_state_indices:
                    update_screwdriver_yaw_wrap_after_recovery(wrap_params, state_t)
                task_state = wrap_screwdriver_task_state_yaw(wrap_params, state_t)
                if original_mask[state_idx]:
                    continue
                missing_likelihood_states_total += 1
                pending_states.append(task_state)
                pending_targets.append((block, state_idx))
                if len(pending_states) >= int(args.likelihood_batch_size):
                    flush_pending()

    flush_pending()

    summary["likelihood_relabeling"] = {
        "enabled": True,
        "model_path": config["model_path"],
        "config_path": config["config_path"],
        "likelihood_num_samples": int(config["likelihood_num_samples"]),
        "sine_cosine": bool(config["sine_cosine"]),
        "T_orig": int(config["T_orig"]),
        "type": str(config["type"]),
        "compile_models": bool(config["compile_models"]),
        "online_likelihood_threshold": ONLINE_LIKELIHOOD_THRESHOLD,
        "scored_states": missing_likelihood_states_filled,
        "original_likelihood_states_preserved": original_likelihood_states_preserved,
        "missing_likelihood_states_filled": missing_likelihood_states_filled,
        "missing_likelihood_states_unfilled": missing_likelihood_states_total - missing_likelihood_states_filled,
        "likelihood_batches": likelihood_batches,
        "likelihood_batch_size": int(args.likelihood_batch_size),
    }


def attach_original_likelihoods(blocks: list[dict[str, Any]]) -> None:
    for block in blocks:
        state_length = int(block["state_length"])
        original = np.full((state_length,), np.nan, dtype=np.float32)
        arrays = block["arrays"]
        if state_length > 0:
            original[0] = np.float32(block["initial_likelihood"])
        if arrays is not None:
            record_likelihoods = arrays.get("record_likelihoods")
            if record_likelihoods is not None:
                limit = min(len(record_likelihoods), max(0, state_length - 1))
                original[1 : 1 + limit] = record_likelihoods[:limit]
            terminal_indices = arrays.get("recovery_terminal_state_indices")
            terminal_likelihoods = block.get("recovery_terminal_likelihoods", [])
            if terminal_indices is not None:
                for state_idx, likelihood in zip(terminal_indices, terminal_likelihoods):
                    state_idx = int(state_idx)
                    if 0 <= state_idx < state_length:
                        original[state_idx] = np.float32(likelihood)
        if state_length > 1:
            original[state_length - 1] = np.float32(block["final_likelihood"])
        block["original_likelihood"] = original
        block["likelihood_originally_labeled_mask"] = np.isfinite(original)


def refresh_endpoint_likelihood_fields(blocks: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    endpoint_diffs = []
    for block in blocks:
        likelihood = block.get("likelihood")
        if likelihood is None or likelihood.size == 0:
            continue
        old_initial = float(block["initial_likelihood"])
        old_final = float(block["final_likelihood"])
        new_initial = float(likelihood[0])
        new_final = float(likelihood[int(block["state_length"]) - 1])
        block["original_initial_likelihood"] = old_initial
        block["original_final_likelihood"] = old_final
        block["initial_likelihood"] = new_initial
        block["final_likelihood"] = new_final
        block["likelihood_delta"] = new_final - new_initial
        block["improved_likelihood"] = math.isfinite(new_initial) and math.isfinite(new_final) and new_final > new_initial
        if math.isfinite(old_initial) and math.isfinite(new_initial):
            endpoint_diffs.append(abs(old_initial - new_initial))
        if math.isfinite(old_final) and math.isfinite(new_final):
            endpoint_diffs.append(abs(old_final - new_final))
    if endpoint_diffs:
        diffs = np.asarray(endpoint_diffs, dtype=np.float64)
        summary["original_endpoint_likelihood_abs_diff"] = {
            "count": int(diffs.size),
            "max": float(np.max(diffs)),
            "mean": float(np.mean(diffs)),
        }
    else:
        summary["original_endpoint_likelihood_abs_diff"] = {"count": 0, "max": None, "mean": None}


def write_blocks_h5(blocks: list[dict[str, Any]], output_path: Path, summary: dict[str, Any]) -> None:
    if not blocks:
        raise ValueError("No recovery blocks found.")
    non_empty_blocks = [block for block in blocks if block["arrays"] is not None]
    if not non_empty_blocks:
        raise ValueError("No non-empty recovery blocks found.")
    num_blocks = len(blocks)
    max_action_length = max(int(block["action_length"]) for block in blocks)
    max_state_length = max_action_length + 1

    q = np.zeros((num_blocks, max_state_length, 3, 4), dtype=np.float32)
    observation = np.zeros((num_blocks, max_state_length, 3), dtype=np.float32)
    action = np.zeros((num_blocks, max_action_length, 3, 4), dtype=np.float32)
    contact_plan = np.zeros((num_blocks, max_action_length, 3), dtype=np.float32)
    contact_state = np.zeros((num_blocks, max_state_length, 3), dtype=np.float32)
    contact_wrenches = np.zeros((num_blocks, max_state_length, 3, 6), dtype=np.float32)
    contact_forces = np.zeros((num_blocks, max_state_length, 3, 3), dtype=np.float32)
    contact_points = np.zeros((num_blocks, max_state_length, 3, 3), dtype=np.float32)
    valid_action_mask = np.zeros((num_blocks, max_action_length), dtype=np.bool_)
    valid_state_mask = np.zeros((num_blocks, max_state_length), dtype=np.bool_)
    likelihood = np.full((num_blocks, max_state_length), np.nan, dtype=np.float32)
    original_likelihood = np.full((num_blocks, max_state_length), np.nan, dtype=np.float32)
    likelihood_originally_labeled_mask = np.zeros((num_blocks, max_state_length), dtype=np.bool_)
    likelihood_recomputed_mask = np.zeros((num_blocks, max_state_length), dtype=np.bool_)

    string_dtype = h5py.string_dtype(encoding="utf-8")
    source_experiment = np.empty((num_blocks,), dtype=object)
    terminal_reason = np.empty((num_blocks,), dtype=object)
    contact_mode = np.empty((num_blocks, max_action_length), dtype=object)
    contact_mode[:] = ""

    int_fields = {
        key: np.zeros((num_blocks,), dtype=np.int64)
        for key in (
            "source_trial_dir",
            "trial_index",
            "block_index_in_trial",
            "start_contact_index",
            "end_contact_index",
            "start_stage_index",
            "end_stage_index",
            "start_episode_num_steps",
            "end_episode_num_steps",
            "trajectory_lengths",
            "action_lengths",
        )
    }
    float_fields = {
        key: np.zeros((num_blocks,), dtype=np.float32)
        for key in (
            "initial_likelihood",
            "final_likelihood",
            "likelihood_delta",
            "original_initial_likelihood",
            "original_final_likelihood",
            "screwdriver_friction",
            "yaw_joint_friction",
        )
    }
    improved_likelihood = np.zeros((num_blocks,), dtype=np.bool_)
    action_int_fields = {
        key: np.zeros((num_blocks, max_action_length), dtype=np.int64)
        for key in ("episode_num_steps", "stage_index")
    }
    recover = np.zeros((num_blocks, max_action_length), dtype=np.bool_)

    for block_idx, block in enumerate(blocks):
        action_length = int(block["action_length"])
        state_length = int(block["state_length"])
        valid_action_mask[block_idx, :action_length] = True
        valid_state_mask[block_idx, :state_length] = True
        for key in int_fields:
            if key == "trajectory_lengths":
                int_fields[key][block_idx] = state_length
            elif key == "action_lengths":
                int_fields[key][block_idx] = action_length
            else:
                int_fields[key][block_idx] = int(block[key])
        for key in float_fields:
            if key in block:
                float_fields[key][block_idx] = np.float32(block[key])
        improved_likelihood[block_idx] = bool(block["improved_likelihood"])
        source_experiment[block_idx] = str(block["source_experiment"])
        terminal_reason[block_idx] = str(block["terminal_reason"])
        likelihood[block_idx, :state_length] = block["likelihood"]
        original_likelihood[block_idx, :state_length] = block["original_likelihood"]
        likelihood_originally_labeled_mask[block_idx, :state_length] = block["likelihood_originally_labeled_mask"]
        likelihood_recomputed_mask[block_idx, :state_length] = block["likelihood_recomputed_mask"]
        arrays = block["arrays"]
        if arrays is None:
            continue
        states = arrays["states"]
        fill_padded(q, block_idx, states[:, :12].reshape(state_length, 3, 4))
        fill_padded(observation, block_idx, states[:, -3:])
        fill_padded(action, block_idx, arrays["actions"].reshape(action_length, 3, 4))
        fill_padded(contact_plan, block_idx, arrays["contact_plan"])
        fill_padded(contact_state, block_idx, arrays["contact_state"])
        fill_padded(contact_wrenches, block_idx, arrays["contact_wrenches"])
        fill_padded(contact_forces, block_idx, arrays["contact_forces"])
        fill_padded(contact_points, block_idx, arrays["contact_points"])
        modes = arrays["contact_modes"]
        contact_mode[block_idx, : len(modes)] = modes
        action_int_fields["episode_num_steps"][block_idx, :action_length] = arrays["episode_num_steps"]
        action_int_fields["stage_index"][block_idx, :action_length] = arrays["stage_index"]
        recover[block_idx, :action_length] = arrays["recover"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5_file:
        h5_file.attrs["schema"] = "proto5_screwdriver_recovery_blocks_v2"
        h5_file.attrs["contact_order"] = "index,middle,thumb"
        h5_file.attrs["row_semantics"] = "one row per recovery block"
        h5_file.attrs["padding"] = "zero padded; use trajectory_lengths/action_lengths and valid_*_mask"
        h5_file.attrs["online_likelihood_threshold"] = ONLINE_LIKELIHOOD_THRESHOLD
        h5_file.attrs["summary_json"] = json.dumps(summary, sort_keys=True)
        h5_file.create_dataset("q", data=q)
        h5_file.create_dataset("observation", data=observation)
        h5_file.create_dataset("action", data=action)
        h5_file.create_dataset("contact_plan", data=contact_plan)
        h5_file.create_dataset("contact_state", data=contact_state)
        h5_file.create_dataset("contact_wrenches", data=contact_wrenches)
        h5_file.create_dataset("contact_forces", data=contact_forces)
        h5_file.create_dataset("contact_points", data=contact_points)
        h5_file.create_dataset("valid_action_mask", data=valid_action_mask)
        h5_file.create_dataset("valid_state_mask", data=valid_state_mask)
        h5_file.create_dataset("likelihood", data=likelihood)
        h5_file.create_dataset("original_likelihood", data=original_likelihood)
        h5_file.create_dataset("likelihood_originally_labeled_mask", data=likelihood_originally_labeled_mask)
        h5_file.create_dataset("likelihood_recomputed_mask", data=likelihood_recomputed_mask)
        h5_file.create_dataset("source_experiment", data=np.asarray(source_experiment, dtype=string_dtype))
        h5_file.create_dataset("terminal_reason", data=np.asarray(terminal_reason, dtype=string_dtype))
        h5_file.create_dataset("contact_mode", data=np.asarray(contact_mode, dtype=string_dtype))
        h5_file.create_dataset("contact_mode_sequence", data=np.asarray(contact_mode, dtype=string_dtype))
        for key, value in int_fields.items():
            h5_file.create_dataset(key, data=value)
        for key, value in float_fields.items():
            h5_file.create_dataset(key, data=value)
        for key, value in action_int_fields.items():
            h5_file.create_dataset(key, data=value)
        h5_file.create_dataset("recover", data=recover)
        h5_file.create_dataset("improved_likelihood", data=improved_likelihood)


def main() -> None:
    args = parse_args()
    blocks, summary = collect_blocks(args)
    attach_original_likelihoods(blocks)
    recompute_block_likelihoods(blocks, args, summary)
    refresh_endpoint_likelihood_fields(blocks, summary)
    summary["total_recovery_blocks"] = len(blocks)
    summary["non_empty_recovery_blocks"] = sum(1 for block in blocks if block["arrays"] is not None)
    summary["improved_likelihood_blocks"] = sum(1 for block in blocks if bool(block["improved_likelihood"]))
    write_blocks_h5(blocks, args.output, summary)
    print(json.dumps({"output": str(args.output), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
