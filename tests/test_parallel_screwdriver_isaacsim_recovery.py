import importlib.util
import pathlib
import sys


_LAUNCHER_PATH = pathlib.Path(__file__).resolve().parents[1] / "examples" / "parallel_screwdriver_isaacsim_recovery.py"
_LAUNCHER_SPEC = importlib.util.spec_from_file_location("parallel_screwdriver_isaacsim_recovery", _LAUNCHER_PATH)
parallel_recovery = importlib.util.module_from_spec(_LAUNCHER_SPEC)
sys.modules[_LAUNCHER_SPEC.name] = parallel_recovery
_LAUNCHER_SPEC.loader.exec_module(parallel_recovery)


def test_split_trial_range_balances_uneven_shards():
    shards = parallel_recovery.split_trial_range(0, 10, 3)

    assert [(shard.start_ind, shard.end_ind) for shard in shards] == [
        (0, 4),
        (4, 7),
        (7, 10),
    ]


def test_split_trial_range_drops_extra_workers():
    shards = parallel_recovery.split_trial_range(5, 7, 8)

    assert [(shard.start_ind, shard.end_ind) for shard in shards] == [
        (5, 6),
        (6, 7),
    ]


def test_worker_slots_repeat_each_device_by_workers_per_gpu():
    slots = parallel_recovery.create_worker_slots(["cuda:0", "cuda:1"], 2)

    assert [(slot.device, slot.slot_index) for slot in slots] == [
        ("cuda:0", 0),
        ("cuda:0", 1),
        ("cuda:1", 0),
        ("cuda:1", 1),
    ]


def test_masked_worker_command_uses_cuda_zero_inside_child(tmp_path):
    command, cuda_visible_devices = parallel_recovery.build_worker_command(
        config_path=tmp_path / "config.yaml",
        experiment_dir=tmp_path / "run",
        shard=parallel_recovery.TrialShard(start_ind=10, end_ind=20),
        slot=parallel_recovery.WorkerSlot(device="cuda:1", slot_index=0),
        forwarded_args=["--hand", "proto5"],
        mask_cuda_visible_devices=True,
        cycle_initial_grasp_dataset=True,
    )

    assert cuda_visible_devices == "1"
    assert command[command.index("--sim_device") + 1] == "cuda:0"
    assert command[command.index("--controller_device") + 1] == "cuda:0"
    assert command[command.index("--start_ind") + 1] == "10"
    assert command[command.index("--end_ind") + 1] == "20"
    assert command[command.index("--cycle_initial_grasp_dataset") + 1] == "true"
    assert "--hand" in command
    assert command[command.index("--hand") + 1] == "proto5"


def test_unmasked_worker_command_preserves_requested_device(tmp_path):
    command, cuda_visible_devices = parallel_recovery.build_worker_command(
        config_path=tmp_path / "config.yaml",
        experiment_dir=tmp_path / "run",
        shard=parallel_recovery.TrialShard(start_ind=0, end_ind=1),
        slot=parallel_recovery.WorkerSlot(device="cuda:2", slot_index=0),
        forwarded_args=[],
        mask_cuda_visible_devices=False,
        cycle_initial_grasp_dataset=False,
    )

    assert cuda_visible_devices is None
    assert command[command.index("--sim_device") + 1] == "cuda:2"
    assert command[command.index("--controller_device") + 1] == "cuda:2"
    assert command[command.index("--cycle_initial_grasp_dataset") + 1] == "false"
