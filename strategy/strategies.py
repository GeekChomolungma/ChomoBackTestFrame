
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
from backtest.metrics import max_drawdown

@dataclass
class ExecutionCfg:
    fee_bps: float = 0.0 # 1.0
    slippage_bps: float = 0.0

@dataclass
class PortfolioCfg:
    initial_capital: float = 10_000.0
    max_leverage: float = 1.0

@dataclass
class BuyHoldResult:
    equity: pd.Series
    total_return: float

def _apply_fees_slippage(mult: float, exec_cfg: ExecutionCfg, roundtrips: int = 1) -> float:
    fee = (exec_cfg.fee_bps + exec_cfg.slippage_bps) / 10_000.0
    cost = (1 - fee) ** (2 * roundtrips)
    return mult * cost

def buy_and_hold(df: pd.DataFrame, cfg: PortfolioCfg, exec_cfg: ExecutionCfg) -> BuyHoldResult:
    prices = df["close"].to_numpy(dtype=np.float64)
    ts = df["endtime"]
    if len(prices) < 2:
        raise ValueError("Need at least 2 prices for buy-and-hold.")
    gross = prices[-1] / prices[0]
    net = _apply_fees_slippage(gross, exec_cfg, 1)
    equity = pd.Series(prices / prices[0] * cfg.initial_capital, index=ts)
    equity *= _apply_fees_slippage(1.0, exec_cfg, 1)
    return BuyHoldResult(equity=equity, total_return=float(net - 1.0))

@dataclass
class Trade:
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    pnl_mult: float
    pnl_mult_net: float
    win: bool

@dataclass
class ChomoLongOnlyResult:
    equity: pd.Series
    trades: List[Trade]
    total_return: float
    win_rate: float
    max_drawdown: float

def chomo_long_only(df: pd.DataFrame, model_wrapper, horizon: int,
                    cfg: PortfolioCfg, exec_cfg: ExecutionCfg, threshold: float = 0.0) -> ChomoLongOnlyResult:
    L = model_wrapper.window_len
    rolling_win = model_wrapper.rolling_win

    prices = df["close"].to_numpy(dtype=np.float64)
    ts = df["endtime"].to_numpy()
    n = len(df)

    equity = np.full(n, np.nan, dtype=np.float64)
    capital = cfg.initial_capital
    equity[:L+rolling_win] = capital
    trades: List[Trade] = []

    model_wrapper.prime_series(df) # prime the entire series once before the loop
    
    i = L + rolling_win
    while i + horizon < n:
        y_hat = model_wrapper.predict_logret(this_index=i)
        if y_hat > threshold:
            entry_idx = i
            exit_idx = i + horizon
            entry_p = prices[entry_idx]
            exit_p = prices[exit_idx]
            gross_mult = exit_p / entry_p
            net_mult = _apply_fees_slippage(gross_mult, exec_cfg, 1)
            capital *= net_mult
            trades.append(Trade(
                entry_ts=int(ts[entry_idx]), exit_ts=int(ts[exit_idx]),
                entry_price=float(entry_p), exit_price=float(exit_p),
                pnl_mult=float(gross_mult), pnl_mult_net=float(net_mult),
                win=net_mult > 1.0
            ))
            equity[entry_idx:exit_idx+1] = capital
            i = exit_idx + 1
        else:
            equity[i] = capital
            i += 1

    equity_series = pd.Series(equity, index=df["endtime"]).ffill()
    total_ret = float(equity_series.iloc[-1] / cfg.initial_capital - 1.0)
    wr = float(sum(t.win for t in trades) / len(trades)) if trades else 0.0
    mdd = float(max_drawdown(equity_series))
    return ChomoLongOnlyResult(equity=equity_series, trades=trades, total_return=total_ret, win_rate=wr, max_drawdown=mdd)
