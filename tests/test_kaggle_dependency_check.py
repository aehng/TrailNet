import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPLAY_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "overnight-20261107"
    / "kernels"
    / "a-replay"
    / "run.py"
)


def load_replay_module():
    spec = importlib.util.spec_from_file_location("trailnet_a_replay", REPLAY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_dependency_check_preserves_torch_without_reloading(monkeypatch, tmp_path):
    import torch

    replay = load_replay_module()
    replay.PROJECT = tmp_path
    replay.WORK = tmp_path
    replay.LOG_PATH = tmp_path / "replay.log"

    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "segmentation_models_pytorch":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    commands = []

    def run_command(args, env=None):
        commands.append(args)
        if "pip" in args:
            return ""
        completed = subprocess.run(args, check=True, capture_output=True, text=True, env=env)
        return completed.stdout.strip()

    monkeypatch.setattr(replay, "run_command", run_command)
    replay.ensure_deps()

    assert len(commands) == 2
    assert "-c" in commands[0]
    assert "torch==" + torch.__version__ in (tmp_path / "torch-constraints.txt").read_text()
    assert "import json, torch" in commands[1][-1]
    assert "importlib.reload" not in commands[1][-1]


def test_run_command_writes_completion_callback(monkeypatch, tmp_path):
    replay = load_replay_module()
    replay.PROJECT = tmp_path
    replay.WORK = tmp_path
    replay.LOG_PATH = tmp_path / "replay.log"

    output = replay.run_command(
        [sys.executable, "-c", "print('child finished')"], timeout_seconds=3
    )

    callbacks = list((tmp_path / "callbacks").glob("*.callback.json"))
    assert "child finished" in output
    assert len(callbacks) == 1
    callback = json.loads(callbacks[0].read_text())
    assert callback["status"] == "complete"
    assert callback["exit_code"] == 0


def test_run_command_timeout_writes_fallback_callback(monkeypatch, tmp_path):
    replay = load_replay_module()
    replay.PROJECT = tmp_path
    replay.WORK = tmp_path
    replay.LOG_PATH = tmp_path / "replay.log"

    try:
        replay.run_command(
            [sys.executable, "-c", "import time; time.sleep(5)"], timeout_seconds=0.1
        )
    except subprocess.TimeoutExpired:
        pass
    else:
        raise AssertionError("stuck child should reach its timed fallback")

    callbacks = list((tmp_path / "callbacks").glob("*.callback.json"))
    assert len(callbacks) == 1
    callback = json.loads(callbacks[0].read_text())
    assert callback["status"] == "timed_out"
    assert callback["exit_code"] is None
    assert callback["timeout_seconds"] == 0.1


def test_run_command_reports_launch_error_after_callback(monkeypatch, tmp_path):
    replay = load_replay_module()
    replay.PROJECT = tmp_path
    replay.WORK = tmp_path
    replay.LOG_PATH = tmp_path / "replay.log"

    try:
        replay.run_command([str(tmp_path / "missing-executable.exe")], timeout_seconds=1)
    except OSError:
        pass
    else:
        raise AssertionError("a child process that cannot launch must fail")

    callbacks = list((tmp_path / "callbacks").glob("*.callback.json"))
    assert len(callbacks) == 1
    callback = json.loads(callbacks[0].read_text())
    assert callback["status"] == "launch_error"
