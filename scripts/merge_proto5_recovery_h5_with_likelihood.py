#!/usr/bin/env python3
"""Merge Proto5 recovery DiffPF H5 files and add OOD likelihood rows."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict, deque
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
DEFAULT_H5_NAME = "proto5_diffpf_training_data.h5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment_dirs",
        nargs="*",
        type=Path,
        default=[Path(path) for path in DEFAULT_EXPERIMENT_DIRS],
        help="Experiment folders containing proto5_diffpf_training_data.h5 and csvgd/trial_*/traj_data.p.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Merged output H5 path. Source H5 files are never modified.",
    )
    parser.add_argument("--h5-name", default=DEFAULT_H5_NAME)
    parser.add_argument(
        "--trials-per-folder",
        type=int,
        default=None,
        help="Keep only the first N trial_index groups from each folder. Omit to merge all rows.",
    )
    parser.add_argument(
        "--max-rows-per-folder",
        type=int,
        default=None,
        help="Optional row cap per folder after trial filtering.",
    )
    parser.add_argument(
        "--allow-missing-likelihood",
        action="store_true",
        help="Write NaN for rows that cannot be matched to a source traj_data.p likelihood.",
    )
    return parser.parse_args()


def decode_h5_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def to_float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    arr = np.asarray(value)
    if arr.size == 0:
        return float("nan")
    value = arr.reshape(-1)[0]
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def record_key_from_mapping(record: dict[str, Any]) -> tuple[int, int, int, str, bool]:
    return (
        int(record.get("trial_index", -1)),
        int(record.get("stage_index", -1)),
        int(record.get("episode_num_steps", -1)),
        str(record.get("contact_mode", "")),
        bool(record.get("recover", False)),
    )


def record_key_from_h5(h5_file: h5py.File, row_idx: int) -> tuple[int, int, int, str, bool]:
    return (
        int(h5_file["trial_index"][row_idx]),
        int(h5_file["stage_index"][row_idx]),
        int(h5_file["episode_num_steps"][row_idx]),
        decode_h5_scalar(h5_file["contact_mode"][row_idx]),
        bool(h5_file["recover"][row_idx]),
    )


def stage_likelihood_from_traj_data(
    traj_data: dict[str, Any],
    record: dict[str, Any],
    stage_seen_counts: dict[int, int],
) -> float:
    if "likelihood" in record:
        return to_float_or_nan(record.get("likelihood"))

    stage_index = int(record.get("stage_index", -1))
    occurrence = stage_seen_counts[stage_index]
    stage_seen_counts[stage_index] += 1

    stage_series = traj_data.get("pre_action_likelihoods", [])
    if not (1 <= stage_index <= len(stage_series)):
        return float("nan")
    values = stage_series[stage_index - 1]
    if not values:
        return float("nan")
    if occurrence < len(values):
        return to_float_or_nan(values[occurrence])
    return float("nan")


def build_likelihood_lookup(experiment_dir: Path) -> dict[tuple[int, int, int, str, bool], deque[float]]:
    lookup: dict[tuple[int, int, int, str, bool], deque[float]] = defaultdict(deque)
    trial_paths = sorted(
        (experiment_dir / "csvgd").glob("trial_*/traj_data.p"),
        key=lambda path: int(path.parent.name.split("_")[-1]),
    )
    for traj_data_path in trial_paths:
        with open(traj_data_path, "rb") as handle:
            traj_data = pickle.load(handle)
        stage_seen_counts: dict[int, int] = defaultdict(int)
        for record in traj_data.get("hri_diffpf_records", []):
            key = record_key_from_mapping(record)
            likelihood = stage_likelihood_from_traj_data(traj_data, record, stage_seen_counts)
            lookup[key].append(likelihood)
    return lookup


def selected_rows_for_file(
    h5_file: h5py.File,
    *,
    trials_per_folder: int | None,
    max_rows_per_folder: int | None,
) -> np.ndarray:
    row_count = int(h5_file["q"].shape[0])
    rows = np.arange(row_count, dtype=np.int64)
    if trials_per_folder is not None:
        trial_indices = np.asarray(h5_file["trial_index"][:], dtype=np.int64)
        keep_trials: list[int] = []
        seen: set[int] = set()
        for trial_index in trial_indices:
            trial_index = int(trial_index)
            if trial_index in seen:
                continue
            seen.add(trial_index)
            keep_trials.append(trial_index)
            if len(keep_trials) >= int(trials_per_folder):
                break
        rows = rows[np.isin(trial_indices, np.asarray(keep_trials, dtype=np.int64))]
    if max_rows_per_folder is not None:
        rows = rows[: int(max_rows_per_folder)]
    return rows


def row_aligned_dataset_names(h5_file: h5py.File) -> list[str]:
    row_count = int(h5_file["q"].shape[0])
    names: list[str] = []
    for name, obj in h5_file.items():
        if not isinstance(obj, h5py.Dataset):
            continue
        if obj.shape and int(obj.shape[0]) == row_count:
            names.append(name)
    return sorted(names)


def read_selected_datasets(h5_file: h5py.File, rows: np.ndarray, dataset_names: list[str]) -> dict[str, np.ndarray]:
    return {name: h5_file[name][rows] for name in dataset_names if name != "likelihood"}


def derive_likelihoods(
    h5_file: h5py.File,
    rows: np.ndarray,
    lookup: dict[tuple[int, int, int, str, bool], deque[float]],
    *,
    allow_missing: bool,
) -> tuple[np.ndarray, int]:
    existing = h5_file["likelihood"][:] if "likelihood" in h5_file else None
    likelihoods = np.empty((len(rows),), dtype=np.float32)
    missing = 0
    for out_idx, row_idx in enumerate(rows):
        key = record_key_from_h5(h5_file, int(row_idx))
        if lookup.get(key):
            likelihoods[out_idx] = np.float32(lookup[key].popleft())
        elif existing is not None:
            likelihoods[out_idx] = np.float32(existing[int(row_idx)])
        else:
            missing += 1
            likelihoods[out_idx] = np.float32(np.nan)
    if missing and not allow_missing:
        raise RuntimeError(
            f"Could not derive likelihood for {missing} selected rows. "
            "Re-run with --allow-missing-likelihood to keep those rows as NaN."
        )
    return likelihoods, missing


def concatenate_chunks(chunks: list[dict[str, np.ndarray]], name: str) -> np.ndarray:
    arrays = [chunk[name] for chunk in chunks if name in chunk]
    if not arrays:
        raise KeyError(name)
    return np.concatenate(arrays, axis=0)


def write_merged_h5(
    output_path: Path,
    chunks: list[dict[str, np.ndarray]],
    dataset_names: list[str],
    attrs: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as out_file:
        for key, value in attrs.items():
            out_file.attrs[key] = value
        out_file.attrs["likelihood_source"] = "traj_data.pre_action_likelihoods aligned by trial/stage/step/contact metadata"
        out_file.attrs["merged_source_count"] = len(chunks)
        for name in dataset_names:
            data = concatenate_chunks(chunks, name)
            if data.dtype.kind in {"O", "S", "U"}:
                string_dtype = h5py.string_dtype(encoding="utf-8")
                data = np.asarray([decode_h5_scalar(value) for value in data], dtype=string_dtype)
                out_file.create_dataset(name, data=data)
            else:
                out_file.create_dataset(name, data=data)


def main() -> None:
    args = parse_args()
    chunks: list[dict[str, np.ndarray]] = []
    dataset_names: set[str] = set()
    first_attrs: dict[str, Any] | None = None
    summary: list[dict[str, Any]] = []

    for experiment_dir in args.experiment_dirs:
        h5_path = experiment_dir / args.h5_name
        if not h5_path.exists():
            raise FileNotFoundError(h5_path)
        lookup = build_likelihood_lookup(experiment_dir)
        with h5py.File(h5_path, "r") as h5_file:
            if first_attrs is None:
                first_attrs = dict(h5_file.attrs)
            names = row_aligned_dataset_names(h5_file)
            rows = selected_rows_for_file(
                h5_file,
                trials_per_folder=args.trials_per_folder,
                max_rows_per_folder=args.max_rows_per_folder,
            )
            chunk = read_selected_datasets(h5_file, rows, names)
            likelihoods, missing = derive_likelihoods(
                h5_file,
                rows,
                lookup,
                allow_missing=args.allow_missing_likelihood,
            )
            source_name_dtype = h5py.string_dtype(encoding="utf-8")
            chunk["likelihood"] = likelihoods
            chunk["source_experiment"] = np.asarray([experiment_dir.name] * len(rows), dtype=source_name_dtype)
            chunk["source_row_index"] = rows.astype(np.int64)
            chunks.append(chunk)
            dataset_names.update(chunk.keys())
            summary.append(
                {
                    "experiment_dir": str(experiment_dir),
                    "input_rows": int(h5_file["q"].shape[0]),
                    "selected_rows": int(len(rows)),
                    "selected_trials": int(len(np.unique(h5_file["trial_index"][rows])) if len(rows) else 0),
                    "missing_likelihood_rows": int(missing),
                }
            )

    if not chunks:
        raise RuntimeError("No input chunks were selected.")
    write_merged_h5(args.output, chunks, sorted(dataset_names), first_attrs or {})
    print(json.dumps({"output": str(args.output), "sources": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
