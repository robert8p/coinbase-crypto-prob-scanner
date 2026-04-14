from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Sequence

from .config import AppConfig
from .demo_data import STABLES

LEVERAGED_KEYWORDS = ("2X", "3X", "BULL", "BEAR", "UP", "DOWN")
ODDITY_KEYWORDS = ("PERP", "INDEX", "VOL", "FIAT", "USDC-USD", "USDT-USD", "EURC")


@dataclass(slots=True)
class UniverseResult:
    diagnostics: dict
    eligible: List[dict]
    selected_for_fetch: List[dict]


class UniverseBuilder:
    def __init__(self, config: AppConfig):
        self.config = config

    def build(
        self,
        products: List[dict],
        currencies: List[dict],
        volume_map: Dict[str, dict],
        *,
        locked_symbols: Sequence[str] | None = None,
        selection_label: str | None = None,
    ) -> UniverseResult:
        currency_type = {c.get("id"): (((c.get("details") or {}).get("type")) or "").lower() for c in currencies}
        excluded = Counter()
        eligible: List[dict] = []
        now = datetime.now(timezone.utc)
        quotes = set(self.config.universe_quotes)

        for p in products:
            pid = p.get("id") or p.get("product_id")
            if not pid or "-" not in pid:
                excluded["malformed_id"] += 1
                continue
            base, quote = pid.split("-", 1)
            quote = p.get("quote_currency") or p.get("quote_currency_id") or quote
            base = p.get("base_currency") or p.get("base_currency_id") or base
            upper_base = str(base).upper()
            upper_pid = str(pid).upper()

            if pid in self.config.exclusion_list:
                excluded["exclusion_list"] += 1
                continue
            if quote not in quotes:
                excluded["unsupported_quote"] += 1
                continue
            if p.get("status") not in {None, "online", "ONLINE"}:
                excluded["inactive_status"] += 1
                continue
            if p.get("trading_disabled"):
                excluded["trading_disabled"] += 1
                continue
            if p.get("auction_mode"):
                excluded["auction_mode"] += 1
                continue
            if self.config.exclude_view_only and p.get("view_only") is True:
                excluded["view_only"] += 1
                continue
            if p.get("cancel_only"):
                excluded["cancel_only"] += 1
                continue
            if p.get("product_type") and str(p.get("product_type")).upper() not in {"SPOT", "UNKNOWN_PRODUCT_TYPE"}:
                excluded["non_spot_type"] += 1
                continue
            if currency_type.get(base) == "fiat":
                excluded["fiat_base"] += 1
                continue
            if self.config.stablecoin_exclusion_enabled and upper_base in STABLES:
                excluded["stablecoin_base"] += 1
                continue
            if any(k in upper_base for k in LEVERAGED_KEYWORDS):
                excluded["leveraged_token"] += 1
                continue
            if (upper_base.startswith("W") and upper_base not in {"WIF"}) or (upper_base.startswith("CB") and upper_base not in {"CBETH"}):
                excluded["wrapped_or_oddity"] += 1
                continue
            if any(k in upper_pid for k in ODDITY_KEYWORDS):
                excluded["wrapped_or_oddity"] += 1
                continue

            created_at = p.get("created_at")
            listing_age_days = None
            if created_at:
                try:
                    created_dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
                    listing_age_days = max(0, int((now - created_dt).total_seconds() // 86400))
                except Exception:
                    listing_age_days = None
            if listing_age_days is not None and listing_age_days < self.config.universe_min_listing_age_days:
                excluded["listing_age"] += 1
                continue

            vol_summary = volume_map.get(pid, {}) or {}
            last = _to_float(vol_summary.get("approximate_quote_24h_price")) or _to_float(p.get("price")) or 0.0
            spot_volume_24h = _to_float(vol_summary.get("spot_volume_24hour")) or _to_float(p.get("volume_24h")) or 0.0
            spot_volume_30d = _to_float(vol_summary.get("spot_volume_30day")) or 0.0
            dollar_volume_24h = last * spot_volume_24h if last and spot_volume_24h else 0.0
            rolling_dollar_volume = last * (spot_volume_30d / 30.0) if last and spot_volume_30d else dollar_volume_24h
            cohort_rank_score = _cohort_rank_score(
                rolling_dollar_volume=float(rolling_dollar_volume or 0.0),
                dollar_volume_24h=float(dollar_volume_24h or 0.0),
                listing_age_days=listing_age_days,
                quote=str(quote),
            )

            eligible.append({
                **p,
                "id": pid,
                "base_currency": base,
                "quote_currency": quote,
                "listing_age_days": listing_age_days,
                "dollar_volume_24h": float(dollar_volume_24h),
                "rolling_dollar_volume": float(rolling_dollar_volume),
                "last": float(last),
                "cohort_rank_score": float(cohort_rank_score),
            })

        ordered = sorted(
            eligible,
            key=lambda x: (x.get("cohort_rank_score", 0.0), x.get("rolling_dollar_volume", 0.0), x["id"]),
            reverse=True,
        )
        low_volume_signal = sum(1 for p in eligible if p.get("rolling_dollar_volume", 0.0) < self.config.universe_min_24h_dollar_volume_usd)

        if locked_symbols:
            selected_for_fetch, missing = self._select_locked_fetch(ordered, locked_symbols)
            selection_mode = selection_label or "trained_cohort_locked"
        else:
            selected_for_fetch = self._select_for_fetch(ordered)
            missing = []
            selection_mode = selection_label or "dynamic"

        diagnostics = {
            "policy": self.config.universe_policy,
            "selection_mode": selection_mode,
            "quotes": list(self.config.universe_quotes),
            "products_seen": len(products),
            "eligible_count": len(eligible),
            "viable_count": len(eligible),
            "selected_for_fetch_count": len(selected_for_fetch),
            "excluded_by_rule": dict(excluded),
            "viability_excluded": {},
            "viability_signals": {
                "below_configured_volume_floor": low_volume_signal,
            },
        }
        if locked_symbols:
            diagnostics.update({
                "trained_cohort_requested_count": len(list(locked_symbols)),
                "trained_cohort_available_count": len(selected_for_fetch),
                "trained_cohort_missing_count": len(missing),
                "trained_cohort_missing_symbols": missing[:20],
            })
            diagnostics["summary"] = (
                f"policy={self.config.universe_policy} mode={selection_mode} "
                f"eligible={len(eligible)} cohort_available={len(selected_for_fetch)} missing={len(missing)}"
            )
        else:
            diagnostics["summary"] = (
                f"policy={self.config.universe_policy} mode={selection_mode} quotes={','.join(self.config.universe_quotes)} "
                f"eligible={len(eligible)} selected_for_fetch={len(selected_for_fetch)}"
            )
        return UniverseResult(diagnostics=diagnostics, eligible=ordered, selected_for_fetch=selected_for_fetch)

    def _select_locked_fetch(self, ordered_eligible: List[dict], locked_symbols: Sequence[str]) -> tuple[List[dict], List[str]]:
        eligible_map = {p["id"]: p for p in ordered_eligible}
        selected: List[dict] = []
        missing: List[str] = []
        seen = set()
        for symbol in locked_symbols:
            sym = str(symbol)
            if sym in seen:
                continue
            seen.add(sym)
            product = eligible_map.get(sym)
            if product is None:
                missing.append(sym)
            else:
                selected.append(product)
        return selected, missing

    def _select_for_fetch(self, ordered_eligible: List[dict]) -> List[dict]:
        policy = self.config.universe_policy.lower()
        if policy == "top_volume":
            out = ordered_eligible[: self.config.universe_top_n]
        elif policy == "both":
            top = ordered_eligible[: self.config.universe_top_n]
            rest = ordered_eligible[self.config.universe_top_n :]
            out = top + rest
        else:  # full_eligible
            out = ordered_eligible
        if self.config.universe_max_products > 0:
            out = out[: self.config.universe_max_products]
        return out


def _cohort_rank_score(*, rolling_dollar_volume: float, dollar_volume_24h: float, listing_age_days: int | None, quote: str) -> float:
    volume_component = math.log10(max(1.0, rolling_dollar_volume)) * 1.00
    daily_component = math.log10(max(1.0, dollar_volume_24h)) * 0.35
    age_component = min(float(listing_age_days or 0), 365.0) / 365.0 * 0.20
    quote_component = 0.05 if str(quote).upper() == "USD" else 0.0
    return volume_component + daily_component + age_component + quote_component


def _to_float(value) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None

