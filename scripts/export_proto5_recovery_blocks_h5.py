#!/usr/bin/env python3
"""Export Proto5 recovery blocks as one H5 row per recovery attempt."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


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


def recovery_contact_blocks(executed_contacts: list[str]) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    contact_idx = 0
    while contact_idx < len(executed_contacts):
        if executed_contacts[contact_idx] == "turn":
            contact_idx += 1
            continue
        start_idx = contact_idx
        while contact_idx < len(executed_contacts) and executed_contacts[contact_idx] != "turn":
            contact_idx += 1
        blocks.append((start_idx, contact_idx - 1))
    return blocks


def block_terminal_reason(executed_contacts: list[str], end_contact_idx: int) -> str:
    if end_contact_idx + 1 >= len(executed_contacts):
        return "episode_end"
    if executed_contacts[end_contact_idx + 1] == "turn":
        return "switch_to_turn"
    return "next_recovery_stage"


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
    for record in block_records:
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
    return {
        "states": np.stack(states, axis=0).astype(np.float32),
        "actions": np.stack(actions, axis=0).astype(np.float32),
        "contact_plan": np.stack(contact_plan, axis=0).astype(np.float32),
        "contact_state": np.stack(contact_state, axis=0).astype(np.float32),
        "contact_wrenches": np.stack(contact_wrenches, axis=0).astype(np.float32),
        "contact_forces": np.stack(contact_forces, axis=0).astype(np.float32),
        "contact_points": np.stack(contact_points, axis=0).astype(np.float32),
        "contact_modes": np.asarray(contact_modes, dtype=object),
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
    blocks: list[dict[str, Any]] = []
    stats = defaultdict(int)
    for block_idx, (start_contact_idx, end_contact_idx) in enumerate(recovery_contact_blocks(executed_contacts)):
        stage_indices = set(range(start_contact_idx + 1, end_contact_idx + 2))
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
        else:
            arrays = None
            action_length = 0
            trial_index = trial_dir_index
            start_step = -1
            end_step = -1
        start_likelihood = previous_likelihood(pre_action_likelihoods, final_likelihoods, start_contact_idx)
        final_likelihood = stage_value(final_likelihoods, end_contact_idx)
        blocks.append(
            {
                "arrays": arrays,
                "action_length": action_length,
                "state_length": action_length + 1,
                "source_experiment": source_experiment,
                "source_trial_dir": trial_dir_index,
                "trial_index": trial_index,
                "block_index_in_trial": block_idx,
                "start_contact_index": start_contact_idx,
                "end_contact_index": end_contact_idx,
                "start_stage_index": start_contact_idx + 1,
                "end_stage_index": end_contact_idx + 1,
                "start_episode_num_steps": start_step,
                "end_episode_num_steps": end_step,
                "initial_likelihood": start_likelihood,
                "final_likelihood": final_likelihood,
                "likelihood_delta": final_likelihood - start_likelihood,
                "improved_likelihood": (
                    (not math.isnan(start_likelihood))
                    and (not math.isnan(final_likelihood))
                    and final_likelihood > start_likelihood
                ),
                "terminal_reason": block_terminal_reason(executed_contacts, end_contact_idx),
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

    string_dtype = h5py.string_dtype(encoding="utf-8")
    source_experiment = np.empty((num_blocks,), dtype=object)
    terminal_reason = np.empty((num_blocks,), dtype=object)
    contact_mode_sequence = np.empty((num_blocks, max_action_length), dtype=object)
    contact_mode_sequence[:] = ""

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
        for key in ("initial_likelihood", "final_likelihood", "likelihood_delta")
    }
    improved_likelihood = np.zeros((num_blocks,), dtype=np.bool_)

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
            float_fields[key][block_idx] = np.float32(block[key])
        improved_likelihood[block_idx] = bool(block["improved_likelihood"])
        source_experiment[block_idx] = str(block["source_experiment"])
        terminal_reason[block_idx] = str(block["terminal_reason"])
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
        contact_mode_sequence[block_idx, : len(modes)] = modes

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5_file:
        h5_file.attrs["schema"] = "proto5_screwdriver_recovery_blocks_v1"
        h5_file.attrs["contact_order"] = "index,middle,thumb"
        h5_file.attrs["row_semantics"] = "one row per recovery block"
        h5_file.attrs["padding"] = "zero padded; use trajectory_lengths/action_lengths and valid_*_mask"
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
        h5_file.create_dataset("source_experiment", data=np.asarray(source_experiment, dtype=string_dtype))
        h5_file.create_dataset("terminal_reason", data=np.asarray(terminal_reason, dtype=string_dtype))
        h5_file.create_dataset(
            "contact_mode_sequence",
            data=np.asarray(contact_mode_sequence, dtype=string_dtype),
        )
        for key, value in int_fields.items():
            h5_file.create_dataset(key, data=value)
        for key, value in float_fields.items():
            h5_file.create_dataset(key, data=value)
        h5_file.create_dataset("improved_likelihood", data=improved_likelihood)


def main() -> None:
    args = parse_args()
    blocks, summary = collect_blocks(args)
    summary["total_recovery_blocks"] = len(blocks)
    summary["non_empty_recovery_blocks"] = sum(1 for block in blocks if block["arrays"] is not None)
    summary["improved_likelihood_blocks"] = sum(1 for block in blocks if bool(block["improved_likelihood"]))
    write_blocks_h5(blocks, args.output, summary)
    print(json.dumps({"output": str(args.output), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
