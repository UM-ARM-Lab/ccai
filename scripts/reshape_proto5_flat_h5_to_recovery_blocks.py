#!/usr/bin/env python3
"""Reshape a flat Proto5 DiffPF training H5 into recovery-block rows."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REQUIRED_DATASETS = (
    "action",
    "contact_forces",
    "contact_mode",
    "contact_plan",
    "contact_points",
    "contact_state",
    "contact_wrenches",
    "episode_num_steps",
    "likelihood",
    "observation",
    "q",
    "q_wrist",
    "recover",
    "robot_joint_pos_full",
    "screwdriver_friction",
    "stage_index",
    "trial_index",
    "yaw_joint_friction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Flat proto5_diffpf_training_data.h5 file.")
    parser.add_argument("--output", type=Path, required=True, help="Output recovery-block H5 path.")
    parser.add_argument(
        "--wait-stable-seconds",
        type=float,
        default=0.0,
        help="Optionally wait until the input file size is unchanged for this many seconds before reading.",
    )
    return parser.parse_args()


def wait_for_stable_file(path: Path, stable_seconds: float) -> None:
    if stable_seconds <= 0:
        return
    previous_size = -1
    stable_since = time.monotonic()
    while True:
        current_size = path.stat().st_size
        now = time.monotonic()
        if current_size != previous_size:
            previous_size = current_size
            stable_since = now
        if now - stable_since >= stable_seconds:
            return
        time.sleep(min(1.0, stable_seconds))


def decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def validate_flat_h5(h5_file: h5py.File) -> int:
    missing = [key for key in REQUIRED_DATASETS if key not in h5_file]
    if missing:
        raise KeyError(f"Missing required datasets: {', '.join(missing)}")
    row_count = int(h5_file["recover"].shape[0])
    for key in REQUIRED_DATASETS:
        if int(h5_file[key].shape[0]) != row_count:
            raise ValueError(f"{key} has {h5_file[key].shape[0]} rows, expected {row_count}.")
    return row_count


def find_recovery_blocks(trial_index: np.ndarray, recover: np.ndarray) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for trial in np.unique(trial_index):
        rows_for_trial = np.flatnonzero(trial_index == trial)
        active_rows: list[int] = []
        for trial_pos, row in enumerate(rows_for_trial):
            row = int(row)
            if bool(recover[row]):
                active_rows.append(row)
                continue
            if active_rows:
                previous_pos = trial_pos - len(active_rows) - 1
                blocks.append(
                    {
                        "rows": active_rows,
                        "terminal_reason": "return_to_id",
                        "terminal_row": row,
                        "previous_row": int(rows_for_trial[previous_pos]) if previous_pos >= 0 else None,
                    }
                )
                active_rows = []
        if active_rows:
            start_pos = int(np.flatnonzero(rows_for_trial == active_rows[0])[0])
            blocks.append(
                {
                    "rows": active_rows,
                    "terminal_reason": "episode_end",
                    "terminal_row": None,
                    "previous_row": int(rows_for_trial[start_pos - 1]) if start_pos > 0 else None,
                }
            )
    return blocks


def state_source_rows(block: dict[str, Any]) -> tuple[int, np.ndarray]:
    rows = np.asarray(block["rows"], dtype=np.int64)
    terminal_row = block["terminal_row"]
    if terminal_row is None:
        return int(rows[0]), rows
    return int(rows[0]), np.asarray([*rows[:-1], int(terminal_row)], dtype=np.int64)


def write_recovery_blocks_h5(input_path: Path, output_path: Path) -> dict[str, Any]:
    with h5py.File(input_path, "r") as src:
        source_rows = validate_flat_h5(src)
        trial_index = src["trial_index"][:]
        recover = src["recover"][:].astype(bool)
        blocks = find_recovery_blocks(trial_index, recover)
        if not blocks:
            raise ValueError(f"No recovery blocks found in {input_path}.")

        action_lengths = np.asarray([len(block["rows"]) for block in blocks], dtype=np.int64)
        num_blocks = int(len(blocks))
        max_action_length = int(action_lengths.max())
        max_state_length = max_action_length + 1

        q = np.zeros((num_blocks, max_state_length, 3, 4), dtype=np.float32)
        q_wrist = np.zeros((num_blocks, max_state_length, 2), dtype=np.float32)
        observation = np.zeros((num_blocks, max_state_length, 3), dtype=np.float32)
        robot_joint_pos_full = np.zeros((num_blocks, max_state_length, 18), dtype=np.float32)
        contact_state = np.zeros((num_blocks, max_state_length, 3), dtype=np.float32)
        contact_wrenches = np.zeros((num_blocks, max_state_length, 3, 6), dtype=np.float32)
        contact_forces = np.zeros((num_blocks, max_state_length, 3, 3), dtype=np.float32)
        contact_points = np.zeros((num_blocks, max_state_length, 3, 3), dtype=np.float32)
        action = np.zeros((num_blocks, max_action_length, 3, 4), dtype=np.float32)
        contact_plan = np.zeros((num_blocks, max_action_length, 3), dtype=np.float32)
        likelihood = np.full((num_blocks, max_state_length), np.nan, dtype=np.float32)
        original_likelihood = np.full((num_blocks, max_state_length), np.nan, dtype=np.float32)
        likelihood_originally_labeled_mask = np.zeros((num_blocks, max_state_length), dtype=np.bool_)
        likelihood_recomputed_mask = np.zeros((num_blocks, max_state_length), dtype=np.bool_)
        valid_action_mask = np.zeros((num_blocks, max_action_length), dtype=np.bool_)
        valid_state_mask = np.zeros((num_blocks, max_state_length), dtype=np.bool_)
        contact_mode = np.empty((num_blocks, max_action_length), dtype=object)
        contact_mode[:] = ""
        terminal_reason = np.empty((num_blocks,), dtype=object)
        recover_out = np.zeros((num_blocks, max_action_length), dtype=np.bool_)
        improved_likelihood = np.zeros((num_blocks,), dtype=np.bool_)
        action_int_fields = {
            key: np.zeros((num_blocks, max_action_length), dtype=np.int64)
            for key in ("episode_num_steps", "stage_index")
        }
        int_fields = {
            key: np.zeros((num_blocks,), dtype=np.int64)
            for key in (
                "trial_index",
                "block_index_in_trial",
                "start_row_index",
                "end_row_index",
                "terminal_row_index",
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
                "screwdriver_friction",
                "yaw_joint_friction",
            )
        }

        block_counts_by_trial: dict[int, int] = {}
        for block_idx, block in enumerate(blocks):
            rows = np.asarray(block["rows"], dtype=np.int64)
            action_length = int(len(rows))
            state_length = action_length + 1
            trial = int(src["trial_index"][rows[0]])
            block_index_in_trial = block_counts_by_trial.get(trial, 0)
            block_counts_by_trial[trial] = block_index_in_trial + 1
            pre_row, post_rows = state_source_rows(block)

            valid_action_mask[block_idx, :action_length] = True
            valid_state_mask[block_idx, :state_length] = True
            for key, dest in (
                ("q", q),
                ("q_wrist", q_wrist),
                ("observation", observation),
                ("robot_joint_pos_full", robot_joint_pos_full),
                ("contact_state", contact_state),
                ("contact_wrenches", contact_wrenches),
                ("contact_forces", contact_forces),
                ("contact_points", contact_points),
            ):
                dest[block_idx, 0] = src[key][pre_row, 0]
                dest[block_idx, 1:state_length] = src[key][post_rows, 1 if block["terminal_row"] is None else 0]
                if block["terminal_row"] is None:
                    continue
                dest[block_idx, 1:action_length] = src[key][rows[:-1], 1]

            action[block_idx, :action_length] = src["action"][rows, 0]
            contact_plan[block_idx, :action_length] = src["contact_plan"][rows, 0]
            contact_mode[block_idx, :action_length] = decode_strings(src["contact_mode"][rows])
            action_int_fields["episode_num_steps"][block_idx, :action_length] = src["episode_num_steps"][rows]
            action_int_fields["stage_index"][block_idx, :action_length] = src["stage_index"][rows]
            recover_out[block_idx, :action_length] = src["recover"][rows]

            previous_row = block["previous_row"]
            terminal_row = block["terminal_row"]
            initial = (
                np.float32(src["likelihood"][previous_row])
                if previous_row is not None
                else np.float32(src["likelihood"][rows[0]])
            )
            final = (
                np.float32(src["likelihood"][terminal_row])
                if terminal_row is not None
                else np.float32(src["likelihood"][rows[-1]])
            )
            likelihood[block_idx, 0] = initial
            if terminal_row is None:
                likelihood[block_idx, 1:state_length] = src["likelihood"][rows].astype(np.float32)
            else:
                likelihood[block_idx, 1:action_length] = src["likelihood"][rows[:-1]].astype(np.float32)
                likelihood[block_idx, action_length] = final
            original_likelihood[block_idx, :state_length] = likelihood[block_idx, :state_length]
            likelihood_originally_labeled_mask[block_idx, :state_length] = np.isfinite(
                likelihood[block_idx, :state_length]
            )

            int_fields["trial_index"][block_idx] = trial
            int_fields["block_index_in_trial"][block_idx] = block_index_in_trial
            int_fields["start_row_index"][block_idx] = int(rows[0])
            int_fields["end_row_index"][block_idx] = int(rows[-1])
            int_fields["terminal_row_index"][block_idx] = int(terminal_row) if terminal_row is not None else -1
            int_fields["start_stage_index"][block_idx] = int(src["stage_index"][rows[0]])
            int_fields["end_stage_index"][block_idx] = int(src["stage_index"][rows[-1]])
            int_fields["start_episode_num_steps"][block_idx] = int(src["episode_num_steps"][rows[0]])
            int_fields["end_episode_num_steps"][block_idx] = int(src["episode_num_steps"][rows[-1]])
            int_fields["trajectory_lengths"][block_idx] = state_length
            int_fields["action_lengths"][block_idx] = action_length
            float_fields["initial_likelihood"][block_idx] = initial
            float_fields["final_likelihood"][block_idx] = final
            float_fields["likelihood_delta"][block_idx] = final - initial
            float_fields["screwdriver_friction"][block_idx] = np.float32(src["screwdriver_friction"][rows[0]])
            float_fields["yaw_joint_friction"][block_idx] = np.float32(src["yaw_joint_friction"][rows[0]])
            improved_likelihood[block_idx] = bool(np.isfinite(initial) and np.isfinite(final) and final > initial)
            terminal_reason[block_idx] = str(block["terminal_reason"])

        terminal_values, terminal_counts = np.unique(np.asarray(terminal_reason, dtype=str), return_counts=True)
        summary = {
            "source_h5": str(input_path),
            "source_rows": int(source_rows),
            "source_trials": int(len(np.unique(trial_index))),
            "total_recovery_blocks": int(num_blocks),
            "valid_action_rows": int(valid_action_mask.sum()),
            "valid_state_rows": int(valid_state_mask.sum()),
            "terminal_reason_counts": {
                str(key): int(value) for key, value in zip(terminal_values, terminal_counts)
            },
            "action_length_min": int(action_lengths.min()),
            "action_length_max": int(action_lengths.max()),
            "action_length_mean": float(action_lengths.mean()),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(output_path, "w") as dest:
            dest.attrs["schema"] = "proto5_screwdriver_recovery_blocks_from_flat_v1"
            dest.attrs["source_h5"] = str(input_path)
            dest.attrs["row_semantics"] = "one row per recovery block from flat transition H5"
            dest.attrs["block_definition"] = (
                "contiguous recover=True rows within a trial; closes on next recover=False row or trial end"
            )
            dest.attrs["padding"] = "zero padded; use trajectory_lengths/action_lengths and valid_*_mask"
            dest.attrs["summary_json"] = json.dumps(summary, sort_keys=True)
            for key, value in (
                ("q", q),
                ("q_wrist", q_wrist),
                ("observation", observation),
                ("robot_joint_pos_full", robot_joint_pos_full),
                ("action", action),
                ("contact_plan", contact_plan),
                ("contact_state", contact_state),
                ("contact_wrenches", contact_wrenches),
                ("contact_forces", contact_forces),
                ("contact_points", contact_points),
                ("valid_action_mask", valid_action_mask),
                ("valid_state_mask", valid_state_mask),
                ("likelihood", likelihood),
                ("original_likelihood", original_likelihood),
                ("likelihood_originally_labeled_mask", likelihood_originally_labeled_mask),
                ("likelihood_recomputed_mask", likelihood_recomputed_mask),
                ("recover", recover_out),
                ("improved_likelihood", improved_likelihood),
            ):
                dest.create_dataset(key, data=value)
            dest.create_dataset("contact_mode", data=np.asarray(contact_mode, dtype=string_dtype))
            dest.create_dataset("contact_mode_sequence", data=np.asarray(contact_mode, dtype=string_dtype))
            dest.create_dataset("terminal_reason", data=np.asarray(terminal_reason, dtype=string_dtype))
            if "robot_joint_names" in src:
                dest.create_dataset("robot_joint_names", data=src["robot_joint_names"][:], dtype=string_dtype)
            for fields in (int_fields, float_fields, action_int_fields):
                for key, value in fields.items():
                    dest.create_dataset(key, data=value)
        return summary


def main() -> None:
    args = parse_args()
    wait_for_stable_file(args.input, float(args.wait_stable_seconds))
    summary = write_recovery_blocks_h5(args.input, args.output)
    print(json.dumps({"output": str(args.output), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
