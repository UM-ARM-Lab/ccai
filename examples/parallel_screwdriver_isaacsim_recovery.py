"""Launch Isaac Sim screwdriver recovery shards across multiple GPU workers."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Sequence

import yaml


CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
SINGLE_WORKER_SCRIPT = pathlib.Path(__file__).resolve().parent / "screwdriver_isaacsim_recovery.py"
DEFAULT_CONFIG_PATH = (
    CCAI_PATH
    / "examples"
    / "config"
    / "proto5"
    / "proto_grey_screwdriver_csvto_TODR_recovery_data_gen.yaml"
)


@dataclass(frozen=True)
class WorkerSlot:
    device: str
    slot_index: int

    @property
    def label(self) -> str:
        sanitized = self.device.replace(":", "_").replace("/", "_")
        return f"{sanitized}_slot{self.slot_index}"


@dataclass(frozen=True)
class TrialShard:
    start_ind: int
    end_ind: int


def _bool_from_cli(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--workers_per_gpu", type=int, default=1)
    parser.add_argument("--start_ind", type=int, default=None)
    parser.add_argument("--end_ind", type=int, default=None)
    parser.add_argument("--experiment_dir", type=pathlib.Path, default=None)
    parser.add_argument("--mask_cuda_visible_devices", type=_bool_from_cli, default=True)
    parser.add_argument("--cycle_initial_grasp_dataset", type=_bool_from_cli, default=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_known_args(argv)


def resolve_config_path(config_path: pathlib.Path) -> pathlib.Path:
    config_path = pathlib.Path(config_path).expanduser()
    if not config_path.is_absolute():
        config_path = CCAI_PATH / config_path
    return config_path


def load_yaml_config(config_path: pathlib.Path) -> dict[str, Any]:
    with open(resolve_config_path(config_path), "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_devices(devices_arg: Sequence[str] | None) -> list[str]:
    if devices_arg:
        return [str(device) for device in devices_arg]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        if tokens:
            return [f"cuda:{token}" for token in tokens]
    try:
        import torch

        count = int(torch.cuda.device_count())
    except Exception:
        count = 0
    if count > 0:
        return [f"cuda:{idx}" for idx in range(count)]
    return ["cuda:0"]


def create_worker_slots(devices: Sequence[str], workers_per_gpu: int) -> list[WorkerSlot]:
    if not devices:
        raise ValueError("At least one device is required.")
    if int(workers_per_gpu) <= 0:
        raise ValueError(f"--workers_per_gpu must be > 0, got {workers_per_gpu}.")
    return [
        WorkerSlot(device=str(device), slot_index=slot_index)
        for device in devices
        for slot_index in range(int(workers_per_gpu))
    ]


def split_trial_range(start_ind: int, end_ind: int, max_shards: int) -> list[TrialShard]:
    if int(end_ind) <= int(start_ind):
        raise ValueError(f"Expected end_ind > start_ind, got {start_ind}..{end_ind}.")
    if int(max_shards) <= 0:
        raise ValueError(f"max_shards must be > 0, got {max_shards}.")
    total = int(end_ind) - int(start_ind)
    num_shards = min(total, int(max_shards))
    base = total // num_shards
    remainder = total % num_shards
    shards = []
    cursor = int(start_ind)
    for shard_index in range(num_shards):
        width = base + (1 if shard_index < remainder else 0)
        shards.append(TrialShard(start_ind=cursor, end_ind=cursor + width))
        cursor += width
    return shards


def cuda_visible_devices_token(device: str) -> str | None:
    normalized = str(device).strip()
    if normalized.startswith("cuda:"):
        return normalized.split(":", 1)[1]
    if normalized.isdigit():
        return normalized
    if normalized == "cuda":
        return "0"
    return None


def child_device_for_slot(device: str, *, mask_cuda_visible_devices: bool) -> str:
    if mask_cuda_visible_devices and cuda_visible_devices_token(device) is not None:
        return "cuda:0"
    normalized = str(device).strip()
    if normalized.isdigit():
        return f"cuda:{normalized}"
    if normalized == "cuda":
        return "cuda:0"
    return normalized


def format_logged_command(command: Sequence[str], cuda_visible_devices: str | None = None) -> str:
    prefix = ""
    if cuda_visible_devices is not None:
        prefix = f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} "
    return prefix + " ".join(str(part) for part in command)


def default_experiment_dir(config: dict[str, Any]) -> pathlib.Path:
    experiment_name = config.get("experiment_name")
    if experiment_name in (None, ""):
        raise ValueError("Recovery config must define experiment_name or --experiment_dir must be provided.")
    suffix = ""
    if bool(config.get("timestamp_experiment", False)):
        suffix = "." + datetime.datetime.now().strftime("%m.%d.%y:%I:%M:%S")
    return CCAI_PATH / "data" / "experiments" / f"{experiment_name}{suffix}"


def resolve_experiment_dir(experiment_dir: pathlib.Path | None, config: dict[str, Any]) -> pathlib.Path:
    if experiment_dir is None:
        return default_experiment_dir(config)
    resolved = pathlib.Path(experiment_dir).expanduser()
    if not resolved.is_absolute():
        resolved = CCAI_PATH / resolved
    return resolved


def build_worker_command(
    *,
    config_path: pathlib.Path,
    experiment_dir: pathlib.Path,
    shard: TrialShard,
    slot: WorkerSlot,
    forwarded_args: Sequence[str],
    mask_cuda_visible_devices: bool,
    cycle_initial_grasp_dataset: bool,
) -> tuple[list[str], str | None]:
    cuda_visible_devices = (
        cuda_visible_devices_token(slot.device)
        if mask_cuda_visible_devices
        else None
    )
    child_device = child_device_for_slot(
        slot.device,
        mask_cuda_visible_devices=mask_cuda_visible_devices,
    )
    command = [
        sys.executable,
        str(SINGLE_WORKER_SCRIPT),
        "--config",
        str(resolve_config_path(config_path)),
        *list(forwarded_args),
        "--start_ind",
        str(int(shard.start_ind)),
        "--end_ind",
        str(int(shard.end_ind)),
        "--experiment_dir",
        str(experiment_dir),
        "--sim_device",
        child_device,
        "--controller_device",
        child_device,
        "--write_pregrasp_states",
        "false",
        "--cycle_initial_grasp_dataset",
        "true" if cycle_initial_grasp_dataset else "false",
    ]
    return command, cuda_visible_devices


def run_worker(
    *,
    command: Sequence[str],
    cuda_visible_devices: str | None,
    log_path: pathlib.Path,
) -> dict[str, Any]:
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    started_at = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(format_logged_command(command, cuda_visible_devices) + "\n\n")
        log_file.flush()
        proc = subprocess.Popen(
            list(command),
            cwd=str(CCAI_PATH),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = proc.wait()
    return {
        "return_code": int(return_code),
        "elapsed_s": time.monotonic() - started_at,
        "log_path": str(log_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args, forwarded_args = parse_args(argv)
    config_path = resolve_config_path(args.config)
    config = load_yaml_config(config_path)
    if str(config.get("mode", "simulation")).lower() != "simulation":
        raise ValueError("Parallel launcher only supports mode: simulation.")

    start_ind = int(args.start_ind if args.start_ind is not None else config.get("start_ind", 0))
    end_ind = int(args.end_ind if args.end_ind is not None else config.get("end_ind", config.get("num_episodes")))
    devices = resolve_devices(args.devices)
    slots = create_worker_slots(devices, int(args.workers_per_gpu))
    shards = split_trial_range(start_ind, end_ind, len(slots))
    experiment_dir = resolve_experiment_dir(args.experiment_dir, config)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    log_dir = experiment_dir / "_parallel_worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    worker_output_dir = experiment_dir / "_parallel_worker_outputs"
    worker_output_dir.mkdir(parents=True, exist_ok=True)

    worker_entries = []
    for worker_index, (slot, shard) in enumerate(zip(slots, shards)):
        worker_experiment_dir = (
            worker_output_dir
            / f"worker_{worker_index:02d}_{slot.label}_{shard.start_ind}_{shard.end_ind}"
        )
        worker_experiment_dir.mkdir(parents=True, exist_ok=True)
        command, cuda_visible_devices = build_worker_command(
            config_path=config_path,
            experiment_dir=worker_experiment_dir,
            shard=shard,
            slot=slot,
            forwarded_args=forwarded_args,
            mask_cuda_visible_devices=bool(args.mask_cuda_visible_devices),
            cycle_initial_grasp_dataset=bool(args.cycle_initial_grasp_dataset),
        )
        log_path = log_dir / f"worker_{worker_index:02d}_{slot.label}_{shard.start_ind}_{shard.end_ind}.log"
        worker_entries.append(
            {
                "worker_index": worker_index,
                "device": slot.device,
                "slot_index": slot.slot_index,
                "start_ind": shard.start_ind,
                "end_ind": shard.end_ind,
                "cuda_visible_devices": cuda_visible_devices,
                "command": command,
                "logged_command": format_logged_command(command, cuda_visible_devices),
                "log_path": str(log_path),
                "experiment_dir": str(worker_experiment_dir),
            }
        )

    manifest = {
        "config_path": str(config_path),
        "experiment_dir": str(experiment_dir),
        "worker_output_dir": str(worker_output_dir),
        "devices": devices,
        "workers_per_gpu": int(args.workers_per_gpu),
        "start_ind": start_ind,
        "end_ind": end_ind,
        "cycle_initial_grasp_dataset": bool(args.cycle_initial_grasp_dataset),
        "mask_cuda_visible_devices": bool(args.mask_cuda_visible_devices),
        "workers": worker_entries,
    }
    manifest_path = experiment_dir / "_parallel_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Experiment directory: {experiment_dir}")
    print(f"Worker count: {len(worker_entries)}")
    print(f"Manifest: {manifest_path}")
    if args.dry_run:
        for entry in worker_entries:
            print(entry["logged_command"])
        return 0

    failures = []
    with ThreadPoolExecutor(max_workers=len(worker_entries)) as executor:
        future_to_entry = {
            executor.submit(
                run_worker,
                command=entry["command"],
                cuda_visible_devices=entry["cuda_visible_devices"],
                log_path=pathlib.Path(entry["log_path"]),
            ): entry
            for entry in worker_entries
        }
        for future in as_completed(future_to_entry):
            entry = future_to_entry[future]
            result = future.result()
            entry.update(result)
            print(
                f"Worker {entry['worker_index']} trials "
                f"{entry['start_ind']}..{entry['end_ind']} exited {result['return_code']} "
                f"after {result['elapsed_s']:.1f}s; log={entry['log_path']}",
                flush=True,
            )
            if int(result["return_code"]) != 0:
                failures.append(entry)

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        print(f"{len(failures)} worker(s) failed. See logs under {log_dir}.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
