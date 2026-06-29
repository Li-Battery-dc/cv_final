"""Minimal helpers for experiment output directories and metadata."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def save_json(path: str, payload: dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)
    return path


def prepare_output_dir(output_root: str,
                       stage_name: str,
                       explicit_output_dir: str | None = None,
                       use_timestamp: bool = True) -> str:
    if explicit_output_dir:
        output_dir = os.path.abspath(explicit_output_dir)
    elif use_timestamp:
        output_root = os.path.abspath(output_root)
        output_dir = os.path.join(output_root, "runs", f"{run_id()}_{stage_name}")
    else:
        output_dir = os.path.abspath(output_root)

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _git_info(cwd: str) -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return None

    commit = run_git(["rev-parse", "HEAD"])
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = run_git(["status", "--short"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_short": status,
    }


def save_run_metadata(output_dir: str,
                      stage: str,
                      params: dict[str, Any] | None = None,
                      inputs: dict[str, Any] | None = None,
                      outputs: dict[str, Any] | None = None,
                      command: list[str] | None = None,
                      env_keys: list[str] | None = None,
                      filename: str = "run_config.json") -> str:
    cwd = os.getcwd()
    env = {}
    for key in env_keys or ["CUDA_VISIBLE_DEVICES", "PYTHONPATH", "VIRTUAL_ENV"]:
        if key in os.environ:
            env[key] = os.environ[key]

    payload = {
        "stage": stage,
        "timestamp_utc": utc_timestamp(),
        "cwd": cwd,
        "command": command or sys.argv,
        "python": sys.executable,
        "git": _git_info(cwd),
        "env": env,
        "params": _json_safe(params or {}),
        "inputs": _json_safe(inputs or {}),
        "outputs": _json_safe(outputs or {"output_dir": os.path.abspath(output_dir)}),
    }
    return save_json(os.path.join(output_dir, filename), payload)
