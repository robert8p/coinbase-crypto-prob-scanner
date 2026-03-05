import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .config import Settings
from .coinbase_client import CoinbaseClient
from .storage import ensure_dir, atomic_write_json, read_json

FALLBACK_PRODUCTS = [
    "BTC-USD","ETH-USD","SOL-USD","XRP-USD","ADA-USD","DOGE-USD","AVAX-USD","LINK-USD","DOT-USD","LTC-USD","BCH-USD",
    "ATOM-USD","UNI-USD","AAVE-USD","ETC-USD","FIL-USD","NEAR-USD","ALGO-USD","HBAR-USD","ARB-USD","OP-USD","INJ-USD",
    "SUI-USD","APT-USD","SEI-USD","IMX-USD","GRT-USD","RNDR-USD","STX-USD",
    "BTC-USDT","ETH-USDT","SOL-USDT","XRP-USDT","ADA-USDT","DOGE-USDT","AVAX-USDT","LINK-USDT","DOT-USDT",
    "BTC-USDC","ETH-USDC","SOL-USDC","XRP-USDC","ADA-USDC","DOGE-USDC",
]

def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

class UniverseManager:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        ensure_dir(Path(cfg.model_dir))
        self.cache_path = Path(cfg.model_dir) / "universe_cache.json"

    def _cache_fresh(self) -> bool:
        if not self.cache_path.exists():
            return False
        try:
            mtime = dt.datetime.fromtimestamp(self.cache_path.stat().st_mtime, tz=dt.timezone.utc)
            age_min = (_now_utc() - mtime).total_seconds()/60.0
            return age_min <= float(self.cfg.universe_refresh_minutes)
        except Exception:
            return False

    def _filter(self, products: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], dict]:
        allow_quotes = set([q.upper() for q in self.cfg.quote_allowlist])
        stable_bases = set([s.upper() for s in self.cfg.stablecoin_bases])
        excluded_stable_base = 0
        quote_dist: dict[str,int] = {}
        out: List[Dict[str, Any]] = []
        for p in products:
            pid = str(p.get("id","")).upper().strip()
            base = str(p.get("base_currency","")).upper().strip()
            quote = str(p.get("quote_currency","")).upper().strip()
            status = str(p.get("status","")).lower().strip()
            if not pid or "-" not in pid:
                continue
            if status and status not in ("online","active"):
                continue
            if quote and quote not in allow_quotes:
                continue
            if self.cfg.exclude_stablecoin_base and base in stable_bases:
                excluded_stable_base += 1
                continue
            quote_dist[quote] = quote_dist.get(quote,0)+1
            out.append({"id": pid, "base": base, "quote": quote, "status": status or "spot"})
        return out, {"quote_distribution": quote_dist, "excluded_stablecoin_base_count": excluded_stable_base}

    async def resolve_universe(self, cb: CoinbaseClient) -> Tuple[List[Dict[str, Any]], dict]:
        cfg = self.cfg
        if cfg.crypto_universe:
            prods = [{"id": s, "base": s.split("-")[0], "quote": s.split("-")[1], "status":"env"} for s in cfg.crypto_universe if "-" in s]
            return (prods if cfg.universe_max <= 0 else prods[:cfg.universe_max]), {"source":"env","count":len(prods)}

        if self._cache_fresh():
            cached = read_json(self.cache_path)
            if cached and isinstance(cached, dict) and "products" in cached:
                prods = cached.get("products") or []
                meta = cached.get("meta") or {}
                meta.update({"source":"cache","count":len(prods)})
                return (prods if cfg.universe_max <= 0 else prods[:cfg.universe_max]), meta

        try:
            products = await cb.list_products()
            filtered, meta = self._filter(products)
            rank = {pid:i for i,pid in enumerate(FALLBACK_PRODUCTS)}
            filtered = sorted(filtered, key=lambda x: (0 if x["id"] in rank else 1, rank.get(x["id"], 1_000_000), x["id"]))
            filtered = filtered if cfg.universe_max <= 0 else filtered[:cfg.universe_max]
            meta.update({"source":"live","count":len(filtered)})
            atomic_write_json(self.cache_path, {"products": filtered, "meta": meta, "updated_at_utc": _now_utc().isoformat()})
            return filtered, meta
        except Exception as e:
            stable = set([s.upper() for s in cfg.stablecoin_bases])
            allow_quotes = set([q.upper() for q in cfg.quote_allowlist])
            fallback = []
            excluded = 0
            for pid in FALLBACK_PRODUCTS:
                base, quote = pid.split("-",1)
                if quote.upper() not in allow_quotes:
                    continue
                if cfg.exclude_stablecoin_base and base.upper() in stable:
                    excluded += 1
                    continue
                fallback.append({"id": pid.upper(), "base": base.upper(), "quote": quote.upper(), "status":"fallback"})
            fallback = fallback if cfg.universe_max <= 0 else fallback[:cfg.universe_max]
            return fallback, {"source":"fallback","count":len(fallback),"excluded_stablecoin_base_count":excluded,"error":f"{type(e).__name__}: {e}"}
