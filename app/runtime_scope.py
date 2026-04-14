from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .persist import atomic_write_json, read_json
from .version import APP_VERSION


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def scope_key(app_version, deployed_since_utc) -> str:
    return f"{app_version or APP_VERSION}|{deployed_since_utc or '-'}"


def _scope_path(model_dir: str | Path) -> Path:
    return Path(model_dir) / "runtime_scope.json"


def _normalize_scope(data: dict | None, *, app_version: str) -> dict | None:
    if not isinstance(data, dict):
        return None
    deployed_since_utc = data.get("deployed_since_utc") or data.get("process_started_at_utc")
    version = data.get("app_version") or app_version
    if version != app_version or not deployed_since_utc:
        return None
    return {
        "app_version": version,
        "deployed_since_utc": deployed_since_utc,
        "process_started_at_utc": data.get("process_started_at_utc") or deployed_since_utc,
        "state_scope_key": data.get("state_scope_key") or scope_key(version, deployed_since_utc),
    }


def initialize_runtime_scope(model_dir: str | Path, *, app_version: str = APP_VERSION, force: bool = False) -> dict:
    path = _scope_path(model_dir)
    existing = _normalize_scope(read_json(path, {}), app_version=app_version)
    if existing is not None and not force:
        return existing
    now = _utc_now_iso()
    payload = {
        "app_version": app_version,
        "deployed_since_utc": now,
        "process_started_at_utc": now,
        "state_scope_key": scope_key(app_version, now),
    }
    atomic_write_json(path, payload)
    return payload


def current_runtime_scope(model_dir: str | Path, *, app_version: str = APP_VERSION) -> dict:
    existing = _normalize_scope(read_json(_scope_path(model_dir), {}), app_version=app_version)
    if existing is not None:
        return existing
    return initialize_runtime_scope(model_dir, app_version=app_version, force=False)
