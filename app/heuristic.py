import math
from typing import Dict

import numpy as np
import pandas as pd

def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0/(1.0+z)
    z = math.exp(x)
    return z/(1.0+z)

def score_heuristic(row: pd.Series) -> Dict[str, float]:
    ret30 = float(row.get("ret_30m",0.0))
    ema = float(row.get("ema_diff_pct",0.0))
    adx = float(row.get("adx",0.0))/50.0
    rvol = float(row.get("rvol_tod",1.0))
    vwap_loc = float(row.get("vwap_loc",0.0))
    donch = float(row.get("donch_dist",0.0))
    atrp = float(row.get("atr_pct",0.0))
    rv = float(row.get("rv",0.0))

    s = 0.0
    s += 5.0 * np.tanh(ret30*8.0)
    s += 3.0 * np.tanh(ema*20.0)
    s += 1.5 * np.tanh(adx)
    s += 1.5 * np.tanh((rvol-1.0)*1.5)
    s += 1.0 * np.tanh(vwap_loc*0.8)
    s -= 0.8 * np.tanh(donch*0.7)
    s -= 1.0 * np.tanh((atrp+rv)*10.0)

    p1 = _sigmoid(s*0.70 - 0.90)
    p2 = _sigmoid(s*0.60 - 1.60)

    p1 = float(min(0.90, max(0.01, p1)))
    p2 = float(min(0.75, max(0.005, p2)))
    return {"prob_1": p1, "prob_2": p2}
