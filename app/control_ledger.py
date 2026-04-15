from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ControlLedgerService:
    def __init__(self, config, *, app_version: str, app_name: str, objective: str):
        self.config = config
        self.app_version = app_version
        self.app_name = app_name
        self.objective = objective
        self.repo_root = Path(__file__).resolve().parent.parent
        self.output_dir = Path(config.model_dir) / "control_ledger"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.facts_json_path = self.output_dir / "latest_control_ledger_facts.json"
        self.facts_txt_path = self.output_dir / "latest_control_ledger_facts.txt"
        self.release_manifest_cache_path = self.output_dir / "latest_release_manifest.json"
        self.release_manifest_txt_path = self.output_dir / "latest_release_manifest.txt"
        self.input_pack_path = self.output_dir / "latest_control_ledger_input_pack.zip"

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _repo_file(self, name: str) -> Path:
        return self.repo_root / name

    def _read_json_file(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _read_text_file(self, path: Path) -> str | None:
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return None

    def _render_text(self, value: Any, indent: int = 0) -> str:
        prefix = " " * indent
        if isinstance(value, dict):
            lines: list[str] = []
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    nested = self._render_text(item, indent + 2)
                    if nested:
                        lines.append(nested)
                else:
                    shown = "-" if item is None else item
                    lines.append(f"{prefix}{key}: {shown}")
            return "\n".join(lines)
        if isinstance(value, list):
            lines = []
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}-")
                    nested = self._render_text(item, indent + 2)
                    if nested:
                        lines.append(nested)
                else:
                    shown = "-" if item is None else item
                    lines.append(f"{prefix}- {shown}")
            return "\n".join(lines)
        return f"{prefix}{'-' if value is None else value}"

    def build_release_manifest(self) -> dict:
        path = self._repo_file("release_manifest.json")
        manifest = self._read_json_file(path)
        if not manifest:
            manifest = {
                "app_name": self.app_name,
                "app_version": self.app_version,
                "tranche_title": "Release manifest missing",
                "tranche_goal": "Populate release_manifest.json in the repo root so the app can expose version-level tranche metadata.",
                "objective": self.objective,
            }
        manifest.setdefault("app_name", self.app_name)
        manifest.setdefault("app_version", self.app_version)
        manifest.setdefault("objective", self.objective)
        manifest["generated_at_utc"] = self._now()
        manifest["source_file"] = "release_manifest.json"
        self.release_manifest_cache_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        self.release_manifest_txt_path.write_text(self._render_text(manifest), encoding="utf-8")
        return manifest

    def _parse_render_metadata(self) -> dict:
        path = self._repo_file("render.yaml")
        raw = self._read_text_file(path)
        if not raw:
            return {"available": False, "path": "render.yaml"}
        lines = raw.splitlines()
        service_name = None
        env = None
        disk_mount = None
        disk_size_gb = None
        env_vars: dict[str, str] = {}
        current_key = None
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("name:") and service_name is None:
                service_name = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("env:") and env is None:
                env = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("mountPath:") and disk_mount is None:
                disk_mount = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("sizeGB:") and disk_size_gb is None:
                disk_size_gb = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("- key:"):
                current_key = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("value:") and current_key:
                env_vars[current_key] = stripped.split(":", 1)[1].strip().strip('"')
                current_key = None
        return {
            "available": True,
            "path": "render.yaml",
            "service_name": service_name,
            "environment": env,
            "disk_mount_path": disk_mount,
            "disk_size_gb": disk_size_gb,
            "env_vars": env_vars,
        }

    def _artifact_inventory(self) -> dict:
        review_dir = Path(self.config.model_dir) / "review_packs"
        replay_dir = Path(self.config.model_dir) / "replay_packs"
        items = {
            "latest_scan_pack": review_dir / "latest_scan_pack.zip",
            "latest_evaluated_pack": review_dir / "latest_evaluated_pack.zip",
            "latest_post_maturity_bundle": Path(self.config.model_dir) / "reviews" / "latest_post_maturity_review_bundle.zip",
            "latest_replay_pack": replay_dir / "latest_replay_pack.zip",
            "model_bundle_pt2": Path(self.config.model_path_pt2),
        }
        out: dict[str, dict[str, Any]] = {}
        for key, path in items.items():
            out[key] = {
                "exists": path.exists(),
                "path": str(path),
                "filename": path.name,
                "modified_at_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat() if path.exists() else None,
                "size_bytes": int(path.stat().st_size) if path.exists() else 0,
            }
        return out

    def build_facts_payload(
        self,
        *,
        status_snapshot: dict,
        health_snapshot: dict,
        current_version_summary: dict,
        automation_status: dict,
        model_output_distribution: dict,
    ) -> dict:
        release_manifest = self.build_release_manifest()
        render_meta = self._parse_render_metadata()
        evidence = (current_version_summary or {}).get("evidence") or {}
        downloads = {
            "facts_json": "/api/control-ledger/facts",
            "facts_txt": "/api/control-ledger/facts.txt",
            "release_manifest_json": "/api/control-ledger/release-manifest",
            "release_manifest_txt": "/api/control-ledger/release-manifest.txt",
            "ledger_input_pack_zip": "/api/control-ledger/ledger-input-pack.zip",
            "status_json": "/api/status",
            "status_txt": "/api/status.txt",
            "health_json": "/health",
            "health_txt": "/health.txt",
            "current_version_summary_json": "/api/reviews/current-version-summary",
            "current_version_summary_txt": "/api/reviews/current-version-summary.txt",
            "current_version_bundle_zip": "/api/reviews/current-version.zip",
            "post_maturity_bundle_zip": "/api/reviews/post-maturity-bundle.zip",
        }
        payload = {
            "generated_at_utc": self._now(),
            "app_identity": {
                "app_name": self.app_name,
                "app_version": self.app_version,
                "objective": self.objective,
                "primary_success_metric": "Visible shortlist beats the hidden remainder for a quality +2.0% move within 240 minutes.",
                "automation_boundary": {
                    "auto_populated_sections": [
                        "app_identity",
                        "current_deployed_state",
                        "current_code_truth",
                        "latest_evidence_reviewed",
                        "artifact_inventory",
                        "download_paths",
                    ],
                    "ai_maintained_sections": [
                        "current_governing_hypothesis",
                        "next_best_step_decision",
                        "explicitly_rejected_or_deprioritized_paths",
                        "open_uncertainties_that_matter_strategically",
                    ],
                    "note": "This payload is factual input for the Control Ledger. Strategic judgment should be maintained by the review/governor chats, not auto-generated by the app.",
                },
            },
            "current_deployed_state": {
                "deployed_version": self.app_version,
                "app_mode": health_snapshot.get("app_mode"),
                "data_source_status": health_snapshot.get("data_source_status"),
                "live_pipeline_mode": status_snapshot.get("live_pipeline_mode"),
                "configured_live_selection_mode": status_snapshot.get("configured_live_selection_mode"),
                "effective_live_selection_mode": status_snapshot.get("effective_live_selection_mode"),
                "effective_live_selection_engine": status_snapshot.get("effective_live_selection_engine"),
                "live_universe_mode_requested": status_snapshot.get("live_universe_mode_requested"),
                "live_universe_mode_effective": status_snapshot.get("live_universe_mode_effective"),
                "target": status_snapshot.get("target") or {},
                "decision_checkpoint": status_snapshot.get("decision_checkpoint") or {},
                "decision_branch_automation": status_snapshot.get("decision_branch_automation") or {},
            },
            "current_code_truth": {
                "repo_root": str(self.repo_root),
                "version_file": "app/version.py",
                "release_manifest_file": "release_manifest.json",
                "render_metadata": render_meta,
                "model_dir": self.config.model_dir,
                "model_bundle_path": self.config.model_path_pt2,
                "readme_present": self._repo_file("README.md").exists(),
                "changelog_present": self._repo_file("CHANGELOG.md").exists(),
            },
            "latest_evidence_reviewed": {
                "current_version_summary": {
                    "generated_at_utc": current_version_summary.get("generated_at_utc"),
                    "scan_pack_count": current_version_summary.get("scan_pack_count"),
                    "evaluated_pack_count": current_version_summary.get("evaluated_pack_count"),
                    "deployed_since_utc": current_version_summary.get("deployed_since_utc"),
                    "headline": evidence.get("headline"),
                    "summary": evidence.get("summary"),
                    "visible_rows": evidence.get("visible_rows"),
                    "non_visible_rows": evidence.get("non_visible_rows"),
                    "resolved_rows": evidence.get("resolved_rows"),
                    "visible_quality_hit_rate": evidence.get("visible_quality_hit_rate"),
                    "non_visible_quality_hit_rate": evidence.get("non_visible_quality_hit_rate"),
                    "visible_avg_end_ret": evidence.get("visible_avg_end_ret"),
                    "non_visible_avg_end_ret": evidence.get("non_visible_avg_end_ret"),
                    "visible_avg_mae": evidence.get("visible_avg_mae"),
                    "non_visible_avg_mae": evidence.get("non_visible_avg_mae"),
                },
                "automation_status": automation_status or {},
                "model_output_distribution": model_output_distribution or {},
            },
            "artifact_inventory": self._artifact_inventory(),
            "download_paths": downloads,
            "release_manifest": release_manifest,
            "operator_handoff_note": {
                "recommended_chat_b_inputs": [
                    "latest_control_ledger_input_pack.zip",
                    "current Control Ledger",
                    "any additional screenshots or deployment notes that are not already in the pack",
                ],
                "recommended_chat_c_inputs": [
                    "updated Control Ledger",
                    "Chat B next-tranche recommendation",
                ],
                "recommended_chat_a_inputs": [
                    "updated Control Ledger",
                    "Chat C governing build prompt",
                    "latest code artifact",
                ],
            },
        }
        self.facts_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self.facts_txt_path.write_text(self._render_text(payload), encoding="utf-8")
        return payload

    def build_input_pack(
        self,
        *,
        status_snapshot: dict,
        health_snapshot: dict,
        current_version_summary: dict,
        automation_status: dict,
        model_output_distribution: dict,
    ) -> Path:
        facts = self.build_facts_payload(
            status_snapshot=status_snapshot,
            health_snapshot=health_snapshot,
            current_version_summary=current_version_summary,
            automation_status=automation_status,
            model_output_distribution=model_output_distribution,
        )
        release_manifest = self.build_release_manifest()
        with zipfile.ZipFile(self.input_pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("control_ledger_facts.json", json.dumps(facts, indent=2, sort_keys=True))
            zf.writestr("control_ledger_facts.txt", self._render_text(facts))
            zf.writestr("release_manifest.json", json.dumps(release_manifest, indent=2, sort_keys=True))
            zf.writestr("release_manifest.txt", self._render_text(release_manifest))
            zf.writestr("api_status_snapshot.json", json.dumps(status_snapshot, indent=2, sort_keys=True, default=str))
            zf.writestr("health_snapshot.json", json.dumps(health_snapshot, indent=2, sort_keys=True, default=str))
            zf.writestr("current_version_summary.json", json.dumps(current_version_summary, indent=2, sort_keys=True, default=str))
            zf.writestr("automation_status.json", json.dumps(automation_status, indent=2, sort_keys=True, default=str))
            zf.writestr("model_output_distribution.json", json.dumps(model_output_distribution, indent=2, sort_keys=True, default=str))
            render_text = self._read_text_file(self._repo_file("render.yaml"))
            if render_text is not None:
                zf.writestr("render.yaml", render_text)
            readme_text = self._read_text_file(self._repo_file("README.md"))
            if readme_text is not None:
                zf.writestr("README.md", readme_text)
            zf.writestr(
                "control_ledger_usage_note.txt",
                "Use control_ledger_facts.json/txt as the factual base for the shared Control Ledger. "
                "Update only the judgment sections in Chat B/Chat C. Do not auto-generate strategic conclusions inside the app.\n"
                "Recommended flow: app pack -> Chat B review -> update Control Ledger -> Chat C governing prompt -> Chat A build.\n",
            )
        return self.input_pack_path

    def build_download_summary(self) -> dict:
        return {
            "available": True,
            "facts_json_path": "/api/control-ledger/facts",
            "facts_txt_path": "/api/control-ledger/facts.txt",
            "release_manifest_json_path": "/api/control-ledger/release-manifest",
            "release_manifest_txt_path": "/api/control-ledger/release-manifest.txt",
            "ledger_input_pack_zip_path": "/api/control-ledger/ledger-input-pack.zip",
            "local_cache": {
                "facts_json": str(self.facts_json_path),
                "facts_txt": str(self.facts_txt_path),
                "release_manifest_json": str(self.release_manifest_cache_path),
                "release_manifest_txt": str(self.release_manifest_txt_path),
                "ledger_input_pack_zip": str(self.input_pack_path),
            },
        }
