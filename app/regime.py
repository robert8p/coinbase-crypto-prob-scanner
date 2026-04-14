from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class MarketRegimeSnapshot:
    state: str
    headline_risk: str
    score: int
    reasons: list[str]
    metrics: dict[str, float]
    override_state: str | None = None
    override_note: str | None = None
    cooldown_active: bool = False
    cooldown_until_utc: str | None = None
    suppress_new_entries: bool = False
    shock_triggered: bool = False
    determined_at_utc: str | None = None
    source: str = "market_regime_engine"
    actionability_state: str = "normal"
    readiness: dict[str, Any] = field(default_factory=dict)
    partial_regime_eligible: bool = False
    regime_publish_warning: bool = False
    regime_publish_warning_reason: str | None = None
    last_partial_publish_attempt_utc: str | None = None
    partial_publish_attempts: int = 0
    partial_publish_successes: int = 0
    partial_publish_failures: int = 0
    last_partial_publish_error: str | None = None
    last_computed_at_utc: str | None = None
    last_applied_at_utc: str | None = None
    last_computed_state: str | None = None
    last_applied_state: str | None = None
    computed_snapshot_version: int = 0
    applied_snapshot_version: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "headline_risk": self.headline_risk,
            "score": self.score,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
            "override_state": self.override_state,
            "override_note": self.override_note,
            "cooldown_active": self.cooldown_active,
            "cooldown_until_utc": self.cooldown_until_utc,
            "suppress_new_entries": self.suppress_new_entries,
            "shock_triggered": self.shock_triggered,
            "determined_at_utc": self.determined_at_utc,
            "source": self.source,
            "actionability_state": self.actionability_state,
            "readiness": dict(self.readiness),
            "partial_regime_eligible": self.partial_regime_eligible,
            "regime_publish_warning": self.regime_publish_warning,
            "regime_publish_warning_reason": self.regime_publish_warning_reason,
            "last_partial_publish_attempt_utc": self.last_partial_publish_attempt_utc,
            "partial_publish_attempts": self.partial_publish_attempts,
            "partial_publish_successes": self.partial_publish_successes,
            "partial_publish_failures": self.partial_publish_failures,
            "last_partial_publish_error": self.last_partial_publish_error,
            "last_computed_at_utc": self.last_computed_at_utc,
            "last_applied_at_utc": self.last_applied_at_utc,
            "last_computed_state": self.last_computed_state,
            "last_applied_state": self.last_applied_state,
            "computed_snapshot_version": self.computed_snapshot_version,
            "applied_snapshot_version": self.applied_snapshot_version,
        }


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _risk_label(state: str) -> str:
    return {"pending": "pending", "green": "low", "amber": "medium", "red": "high"}.get(state, "pending")


def _actionability_state(state: str, *, suppress_new_entries: bool, cooldown_active: bool = False) -> str:
    state = str(state or "pending").lower()
    if state == "pending":
        return "pending_blocked"
    if suppress_new_entries:
        return "blocked"
    if cooldown_active:
        return "cooldown_restricted"
    if state == "amber":
        return "restricted"
    return "normal"


def _ctx_ready(ctx: dict | None) -> bool:
    ctx = ctx or {}
    if bool(ctx.get("_ready")):
        return True
    required = ("ret_15m", "ret_1h", "ret_4h", "rv_ratio_1h_24h")
    return all(k in ctx for k in required)


def assess_market_regime_readiness(config, btc_ctx: dict | None, eth_ctx: dict | None, feature_rows: dict[str, dict] | None) -> dict[str, Any]:
    feature_rows = feature_rows or {}
    observed = len(feature_rows)
    required = max(1, int(getattr(config, "market_regime_partial_min_feature_rows", 80)))
    btc_ready = _ctx_ready(btc_ctx)
    eth_ready = _ctx_ready(eth_ctx)
    breadth_ready = observed >= required
    waiting_on: list[str] = []
    if observed < required:
        waiting_on.append("feature_rows")
    if not btc_ready:
        waiting_on.append("btc_context")
    if not eth_ready:
        waiting_on.append("eth_context")
    if not breadth_ready:
        waiting_on.append("breadth")
    eligible = observed >= required and btc_ready and eth_ready and breadth_ready
    return {
        "partial_regime_eligible": bool(eligible),
        "min_feature_rows_required": required,
        "min_feature_rows_observed": observed,
        "btc_ready": btc_ready,
        "eth_ready": eth_ready,
        "breadth_ready": breadth_ready,
        "waiting_on": waiting_on,
    }


def _publish_meta(previous: dict | None, publish_meta: dict | None = None) -> dict[str, Any]:
    previous = previous or {}
    publish_meta = publish_meta or {}
    return {
        "last_partial_publish_attempt_utc": publish_meta.get("last_partial_publish_attempt_utc") or previous.get("last_partial_publish_attempt_utc"),
        "partial_publish_attempts": int(publish_meta.get("partial_publish_attempts", previous.get("partial_publish_attempts", 0)) or 0),
        "partial_publish_successes": int(publish_meta.get("partial_publish_successes", previous.get("partial_publish_successes", 0)) or 0),
        "partial_publish_failures": int(publish_meta.get("partial_publish_failures", previous.get("partial_publish_failures", 0)) or 0),
        "last_partial_publish_error": publish_meta.get("last_partial_publish_error") if "last_partial_publish_error" in publish_meta else previous.get("last_partial_publish_error"),
        "regime_publish_warning": bool(publish_meta.get("regime_publish_warning", previous.get("regime_publish_warning", False))),
        "regime_publish_warning_reason": publish_meta.get("regime_publish_warning_reason") or previous.get("regime_publish_warning_reason"),
    }


def _lifecycle_meta(previous: dict | None, publish_meta: dict | None = None) -> dict[str, Any]:
    previous = previous or {}
    publish_meta = publish_meta or {}
    return {
        "last_computed_at_utc": publish_meta.get("last_computed_at_utc") or previous.get("last_computed_at_utc"),
        "last_applied_at_utc": publish_meta.get("last_applied_at_utc") or previous.get("last_applied_at_utc"),
        "last_computed_state": publish_meta.get("last_computed_state") or previous.get("last_computed_state"),
        "last_applied_state": publish_meta.get("last_applied_state") or previous.get("last_applied_state"),
        "computed_snapshot_version": int(publish_meta.get("computed_snapshot_version", previous.get("computed_snapshot_version", 0)) or 0),
        "applied_snapshot_version": int(publish_meta.get("applied_snapshot_version", previous.get("applied_snapshot_version", 0)) or 0),
    }


def mark_market_regime_applied(snapshot: MarketRegimeSnapshot, previous: dict | None = None, applied_at_utc: str | None = None) -> MarketRegimeSnapshot:
    previous = previous or {}
    applied_at_utc = applied_at_utc or _utc_now().isoformat()
    prior_applied_version = int(previous.get("applied_snapshot_version", 0) or 0)
    computed_version = int(getattr(snapshot, "computed_snapshot_version", 0) or 0)
    applied_version = max(computed_version, prior_applied_version + 1)
    snapshot.last_applied_at_utc = applied_at_utc
    snapshot.last_applied_state = snapshot.state
    snapshot.applied_snapshot_version = applied_version
    if snapshot.state != "pending" and not snapshot.determined_at_utc:
        snapshot.determined_at_utc = applied_at_utc
    return snapshot


def pending_market_regime(
    previous: dict | None = None,
    reason: str = "regime evaluation pending",
    readiness: dict[str, Any] | None = None,
    publish_meta: dict[str, Any] | None = None,
) -> MarketRegimeSnapshot:
    previous = previous or {}
    readiness = readiness or previous.get("readiness") or {
        "partial_regime_eligible": False,
        "min_feature_rows_required": 0,
        "min_feature_rows_observed": 0,
        "btc_ready": False,
        "eth_ready": False,
        "breadth_ready": False,
        "waiting_on": ["feature_rows", "btc_context", "eth_context", "breadth"],
    }
    meta = _publish_meta(previous, publish_meta)
    lifecycle = _lifecycle_meta(previous, publish_meta)
    if publish_meta and meta["partial_publish_successes"] > 0 and not meta["regime_publish_warning"]:
        meta["regime_publish_warning"] = True
        meta["regime_publish_warning_reason"] = meta["regime_publish_warning_reason"] or "computed_or_counted_but_not_applied"
    cooldown_until = previous.get("cooldown_until_utc")
    cooldown_active = bool(_parse_utc(cooldown_until) and _parse_utc(cooldown_until) > _utc_now())
    return MarketRegimeSnapshot(
        state="pending",
        headline_risk="pending",
        score=0,
        reasons=[reason],
        metrics={},
        override_state=previous.get("override_state"),
        override_note=previous.get("override_note"),
        cooldown_active=cooldown_active,
        cooldown_until_utc=cooldown_until if cooldown_active else None,
        suppress_new_entries=True,
        shock_triggered=False,
        determined_at_utc=None,
        source="market_regime_engine",
        actionability_state="pending_blocked",
        readiness=readiness,
        partial_regime_eligible=bool(readiness.get("partial_regime_eligible")),
        regime_publish_warning=bool(meta["regime_publish_warning"]),
        regime_publish_warning_reason=meta["regime_publish_warning_reason"],
        last_partial_publish_attempt_utc=meta["last_partial_publish_attempt_utc"],
        partial_publish_attempts=meta["partial_publish_attempts"],
        partial_publish_successes=meta["partial_publish_successes"],
        partial_publish_failures=meta["partial_publish_failures"],
    )


def build_market_regime(
    config,
    btc_ctx: dict | None,
    eth_ctx: dict | None,
    feature_rows: dict[str, dict] | None,
    previous: dict | None = None,
    readiness: dict[str, Any] | None = None,
    publish_meta: dict[str, Any] | None = None,
) -> MarketRegimeSnapshot:
    btc_ctx = btc_ctx or {}
    eth_ctx = eth_ctx or {}
    feature_rows = feature_rows or {}
    rows = list(feature_rows.values())
    readiness = readiness or assess_market_regime_readiness(config, btc_ctx, eth_ctx, feature_rows)
    meta = _publish_meta(previous, publish_meta)
    lifecycle = _lifecycle_meta(previous, publish_meta)

    reasons: list[str] = []
    score = 0
    shock_triggered = False

    btc_ret_15m = _safe_float(btc_ctx.get("ret_15m"))
    btc_ret_1h = _safe_float(btc_ctx.get("ret_1h"))
    btc_ret_4h = _safe_float(btc_ctx.get("ret_4h"))
    eth_ret_1h = _safe_float(eth_ctx.get("ret_1h"))
    btc_abs_15m = abs(btc_ret_15m)
    btc_abs_1h = abs(btc_ret_1h)
    btc_abs_4h = abs(btc_ret_4h)
    eth_abs_1h = abs(eth_ret_1h)
    btc_vol_ratio = _safe_float(btc_ctx.get("rv_ratio_1h_24h"), 1.0)
    eth_vol_ratio = _safe_float(eth_ctx.get("rv_ratio_1h_24h"), 1.0)

    breadth_neg_15m = 0.0
    breadth_neg_1h = 0.0
    breadth_abs_15m = 0.0
    breadth_abs_1h = 0.0
    if rows:
        breadth_neg_15m = sum(1 for r in rows if _safe_float(r.get("ret_15m")) < 0) / len(rows)
        breadth_neg_1h = sum(1 for r in rows if _safe_float(r.get("ret_60m")) < 0) / len(rows)
        breadth_abs_15m = sum(1 for r in rows if abs(_safe_float(r.get("ret_15m"))) >= config.market_regime_amber_btc_15m_move) / len(rows)
        breadth_abs_1h = sum(1 for r in rows if abs(_safe_float(r.get("ret_60m"))) >= config.market_regime_amber_btc_1h_move) / len(rows)

    override_state = (getattr(config, "market_regime_override", "") or "").strip().lower() or None
    override_note = (getattr(config, "market_regime_override_note", "") or "").strip() or None

    metrics = {
        "btc_ret_15m": round(btc_ret_15m, 4),
        "btc_ret_1h": round(btc_ret_1h, 4),
        "btc_ret_4h": round(btc_ret_4h, 4),
        "eth_ret_1h": round(eth_ret_1h, 4),
        "btc_abs_15m": round(btc_abs_15m, 4),
        "btc_abs_1h": round(btc_abs_1h, 4),
        "btc_abs_4h": round(btc_abs_4h, 4),
        "eth_abs_1h": round(eth_abs_1h, 4),
        "btc_vol_ratio_1h_24h": round(btc_vol_ratio, 4),
        "eth_vol_ratio_1h_24h": round(eth_vol_ratio, 4),
        "breadth_neg_15m": round(breadth_neg_15m, 4),
        "breadth_neg_1h": round(breadth_neg_1h, 4),
        "breadth_abs_15m": round(breadth_abs_15m, 4),
        "breadth_abs_1h": round(breadth_abs_1h, 4),
    }

    if override_state in {"green", "amber", "red"}:
        now = _utc_now().isoformat()
        cooldown_until = None
        if override_state == "red":
            cooldown_until = (_utc_now() + timedelta(minutes=int(config.market_regime_cooldown_minutes))).isoformat()
        suppress_new_entries = override_state == "red"
        cooldown_active = override_state == "red"
        return MarketRegimeSnapshot(
            state=override_state,
            headline_risk=_risk_label(override_state),
            score=999,
            reasons=[f"operator override: {override_state}"] + ([override_note] if override_note else []),
            metrics=metrics,
            override_state=override_state,
            override_note=override_note,
            cooldown_active=cooldown_active,
            cooldown_until_utc=cooldown_until,
            suppress_new_entries=suppress_new_entries,
            shock_triggered=(override_state == "red"),
            determined_at_utc=now,
            source="operator_override",
            actionability_state=_actionability_state(override_state, suppress_new_entries=suppress_new_entries, cooldown_active=cooldown_active),
            readiness=readiness,
            partial_regime_eligible=bool(readiness.get("partial_regime_eligible")),
            regime_publish_warning=False,
            regime_publish_warning_reason=None,
            last_partial_publish_attempt_utc=meta["last_partial_publish_attempt_utc"],
            partial_publish_attempts=meta["partial_publish_attempts"],
            partial_publish_successes=meta["partial_publish_successes"],
            partial_publish_failures=meta["partial_publish_failures"],
            last_partial_publish_error=meta["last_partial_publish_error"],
            last_computed_at_utc=now,
            last_applied_at_utc=lifecycle["last_applied_at_utc"],
            last_computed_state=override_state,
            last_applied_state=lifecycle["last_applied_state"],
            computed_snapshot_version=int(lifecycle["computed_snapshot_version"] or 0) + 1,
            applied_snapshot_version=lifecycle["applied_snapshot_version"],
        )

    btc_shock_confirmed = (
        btc_ret_15m <= -abs(config.market_regime_red_btc_15m_shock)
        and (
            eth_ret_1h <= -abs(config.market_regime_amber_eth_1h_move)
            or breadth_neg_15m >= config.market_regime_amber_breadth_neg_15m
        )
    )
    if btc_shock_confirmed:
        score += config.market_regime_red_score
        shock_triggered = True
        reasons.append("BTC shock confirmed across market breadth")
    if btc_ret_1h <= -abs(config.market_regime_red_btc_1h_move):
        score += config.market_regime_red_score
        shock_triggered = True
        reasons.append("BTC 1h downside extreme")
    if eth_ret_1h <= -abs(config.market_regime_red_eth_1h_move):
        score += config.market_regime_red_score
        shock_triggered = True
        reasons.append("ETH 1h downside extreme")
    if btc_vol_ratio >= config.market_regime_red_btc_vol_ratio:
        score += config.market_regime_red_score
        shock_triggered = True
        reasons.append("BTC realised vol shock")
    if eth_vol_ratio >= config.market_regime_red_eth_vol_ratio:
        score += config.market_regime_red_score
        shock_triggered = True
        reasons.append("ETH realised vol shock")
    if breadth_neg_15m >= config.market_regime_red_breadth_neg_15m:
        score += config.market_regime_red_score
        shock_triggered = True
        reasons.append("15m breadth deterioration")
    if breadth_neg_1h >= config.market_regime_red_breadth_neg_1h:
        score += config.market_regime_red_score
        shock_triggered = True
        reasons.append("1h breadth deterioration")

    if btc_ret_15m <= -abs(config.market_regime_amber_btc_15m_move):
        score += config.market_regime_amber_score
        reasons.append("BTC 15m downside elevated")
    if btc_ret_1h <= -abs(config.market_regime_amber_btc_1h_move):
        score += config.market_regime_amber_score
        reasons.append("BTC 1h downside elevated")
    if eth_ret_1h <= -abs(config.market_regime_amber_eth_1h_move):
        score += config.market_regime_amber_score
        reasons.append("ETH 1h downside elevated")
    if btc_vol_ratio >= config.market_regime_amber_btc_vol_ratio:
        score += config.market_regime_amber_score
        reasons.append("BTC realised vol elevated")
    if eth_vol_ratio >= config.market_regime_amber_eth_vol_ratio:
        score += config.market_regime_amber_score
        reasons.append("ETH realised vol elevated")
    if breadth_neg_15m >= config.market_regime_amber_breadth_neg_15m:
        score += config.market_regime_amber_score
        reasons.append("15m breadth soft")
    if breadth_neg_1h >= config.market_regime_amber_breadth_neg_1h:
        score += config.market_regime_amber_score
        reasons.append("1h breadth soft")
    if breadth_abs_15m >= 0.65 and breadth_neg_15m >= 0.50:
        score += config.market_regime_amber_score
        reasons.append("broad downside intraday volatility")
    if btc_ret_4h <= -0.04:
        score += config.market_regime_amber_score
        reasons.append("BTC 4h downside elevated")

    if shock_triggered or score >= config.market_regime_red_total_score:
        state = "red"
    elif score >= config.market_regime_amber_total_score:
        state = "amber"
    else:
        state = "green"

    now_dt = _utc_now()
    previous = previous or {}
    prior_until = _parse_utc(previous.get("cooldown_until_utc"))
    cooldown_until = prior_until if prior_until and prior_until > now_dt else None
    if state == "red":
        new_until = now_dt + timedelta(minutes=int(config.market_regime_cooldown_minutes))
        cooldown_until = max(cooldown_until, new_until) if cooldown_until else new_until

    cooldown_active = bool(cooldown_until and cooldown_until > now_dt)
    suppress_new_entries = state == "red"

    if cooldown_active and state == "green":
        state = "amber"
        reasons.append("cooldown still active")

    if not reasons:
        reasons.append("normal market structure")

    return MarketRegimeSnapshot(
        state=state,
        headline_risk=_risk_label(state),
        score=int(score),
        reasons=reasons,
        metrics=metrics,
        override_state=None,
        override_note=override_note,
        cooldown_active=cooldown_active,
        cooldown_until_utc=cooldown_until.isoformat() if cooldown_until else None,
        suppress_new_entries=suppress_new_entries,
        shock_triggered=shock_triggered,
        determined_at_utc=now_dt.isoformat(),
        actionability_state=_actionability_state(state, suppress_new_entries=suppress_new_entries, cooldown_active=cooldown_active),
        readiness=readiness,
        partial_regime_eligible=bool(readiness.get("partial_regime_eligible")),
        regime_publish_warning=False,
        regime_publish_warning_reason=None,
        last_partial_publish_attempt_utc=meta["last_partial_publish_attempt_utc"],
        partial_publish_attempts=meta["partial_publish_attempts"],
        partial_publish_successes=meta["partial_publish_successes"],
        partial_publish_failures=meta["partial_publish_failures"],
        last_partial_publish_error=meta["last_partial_publish_error"],
        last_computed_at_utc=now_dt.isoformat(),
        last_applied_at_utc=lifecycle["last_applied_at_utc"],
        last_computed_state=state,
        last_applied_state=lifecycle["last_applied_state"],
        computed_snapshot_version=int(lifecycle["computed_snapshot_version"] or 0) + 1,
        applied_snapshot_version=lifecycle["applied_snapshot_version"],
    )


def classify_liquidity_tier(symbol: str, diag: dict | None, config) -> str:
    symbol = str(symbol or "")
    majors = set(getattr(config, "market_regime_liquid_major_symbols", []) or [])
    if symbol in {"BTC-USD", "ETH-USD"}:
        return "tier1"
    dv = _safe_float((diag or {}).get("rolling_dollar_volume"))
    if symbol in majors or dv >= float(getattr(config, "market_regime_tier2_volume_floor", 5_000_000.0)):
        return "tier2"
    return "tier3"


def live_policy_for(regime_state: str, tier: str, config) -> dict[str, Any]:
    state = str(regime_state or "green").lower()
    if state == "pending":
        state = "red"
    tier = str(tier or "tier3").lower()
    prefix = f"market_regime_{state}_{tier}"
    return {
        "factor": float(getattr(config, f"{prefix}_factor")),
        "cap": float(getattr(config, f"{prefix}_cap")),
        "threshold": float(getattr(config, f"{prefix}_threshold")),
        "suppress": bool(getattr(config, f"{prefix}_suppress")),
    }
