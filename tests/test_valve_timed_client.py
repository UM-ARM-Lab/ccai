import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

spec = importlib.util.spec_from_file_location('valve_timed_client', Path(__file__).parents[1]/'hardware/valve_timed_client.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_client(monkeypatch, states=(1,3,3), fresh=True):
    obj = module.ValveTimedClient.__new__(module.ValveTimedClient)
    obj.timeout_s = .1
    obj.node = SimpleNamespace(_joint_state_seq=0, current_joint_pose=SimpleNamespace(header=SimpleNamespace(stamp=0)))
    obj.rospy = SimpleNamespace(is_shutdown=lambda: False)
    obj.goal_type = SimpleNamespace
    obj.client = Mock()
    obj.client.get_state.side_effect = list(states)
    obj.client.get_result.return_value = SimpleNamespace(elapsed_s=1/3, completed_at=1)
    clock = [0.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    def advance(_):
        clock[0] += .01
        if fresh:
            obj.node._joint_state_seq += 1
            obj.node.current_joint_pose.header.stamp = 2
    monkeypatch.setattr(module.time, 'sleep', advance)
    return obj


def test_completion_then_fresh_joint_sample(monkeypatch):
    client = make_client(monkeypatch)
    result = client.execute([0.]*16, 40, 1/3)
    assert result.completed_at == 1
    assert client.node._joint_state_seq >= 2
    assert client.client.send_goal.call_count == 1
    assert client.client.send_goal.call_args.args[0].repeats == 40


def test_stale_joint_samples_fail(monkeypatch):
    client = make_client(monkeypatch, states=(3,3), fresh=False)
    with pytest.raises(RuntimeError, match='fresh'):
        client.execute([0.]*16, 40, 1/3)


def test_preempted_action_is_not_success(monkeypatch):
    client = make_client(monkeypatch, states=(2,2))
    client.client.get_goal_status_text.return_value = 'preempted'
    with pytest.raises(RuntimeError, match='interrupted'):
        client.execute([0.]*16, 40, 1/3)


def test_deadline_timeout_cancels(monkeypatch):
    client = make_client(monkeypatch)
    client.client.get_state.side_effect = None
    client.client.get_state.return_value = 1
    with pytest.raises(RuntimeError, match='timed out'):
        client.execute([0.]*16, 40, 1/3)
    client.client.cancel_goal.assert_called_once()


@pytest.mark.parametrize('target,repeats,duration', [([0.]*15,40,1/3), ([float('nan')]*16,40,1/3), ([0.]*16,0,1/3), ([0.]*16,40,0)])
def test_invalid_goals_never_publish(monkeypatch, target, repeats, duration):
    client = make_client(monkeypatch)
    with pytest.raises(ValueError):
        client.execute(target,repeats,duration)
    client.client.send_goal.assert_not_called()
