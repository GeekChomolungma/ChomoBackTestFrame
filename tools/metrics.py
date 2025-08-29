
from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd
from .time_utils import annualization_factor

def max_drawdown(equity: pd.Series) -> float:
    cummax = equity.cummax()
    dd = (equity - cummax) / cummax
    return float(dd.min()) if len(dd) else 0.0

def sharpe_ratio(equity: pd.Series, interval: str, eps: float=1e-12) -> float:
    if len(equity) < 3: return 0.0
    rets = equity.pct_change().dropna()
    if rets.std() < eps: return 0.0
    af = annualization_factor(interval)
    return float(rets.mean() / rets.std() * af)

def calmar_ratio(equity: pd.Series, interval: str) -> float:
    # simple annualized return over |MDD|
    if len(equity) < 2: return 0.0
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
    # crude annualization by sqrt-time (consistent with Sharpe scaling here)
    annual_ret = total_ret * annualization_factor(interval)
    mdd = abs(max_drawdown(equity)) or 1e-12
    return float(annual_ret / mdd)

def win_rate(wins: Iterable[bool]) -> float:
    wins = list(wins)
    return sum(wins)/len(wins) if wins else 0.0
