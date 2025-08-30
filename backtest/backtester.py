
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from iowrapper.mongo_shell import MongoShell
from iowrapper.data_adapters import MongoKlineSource
from model.model_wrapper import ChomoModelWrapper
from strategy.strategies import PortfolioCfg, ExecutionCfg, buy_and_hold, chomo_long_only
from backtest.metrics import sharpe_ratio, calmar_ratio

@dataclass
class RunConfig:
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "market_info"
    mongo_coll: str = "klines"
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    start_ms: int | None = None
    end_ms: int | None = None
    ckpt_path: str = "chomo_ckpt.pt"
    device: str | None = None
    fee_bps: float = 1.0
    slippage_bps: float = 0.0
    initial_capital: float = 10_000.0
    threshold: float = 0.0
    csv_path: str | None = None   # if provided, use CSV source instead of Mongo

def fetch_klines(cfg: RunConfig) -> pd.DataFrame:
    mongo = MongoShell(cfg.mongo_uri, cfg.mongo_db)
    ds = MongoKlineSource(mongo, cfg.mongo_db, cfg.mongo_coll, cfg.symbol, cfg.interval, cfg.start_ms, cfg.end_ms)
    return ds.fetch()

def run_backtest(cfg: RunConfig):
    df = fetch_klines(cfg)
    model = ChomoModelWrapper(cfg.ckpt_path, device=cfg.device)
    port = PortfolioCfg(initial_capital=cfg.initial_capital)
    exe = ExecutionCfg(fee_bps=cfg.fee_bps, slippage_bps=cfg.slippage_bps)

    # Buy & Hold
    bah = buy_and_hold(df, port, exe)

    # Chomo
    res = chomo_long_only(df, model, model.horizon, port, exe, threshold=cfg.threshold)

    # Metrics
    sharpe = sharpe_ratio(res.equity, cfg.interval)
    calmar = calmar_ratio(res.equity, cfg.interval)

    return {
        "klines": df,
        "buy_and_hold": {
            "total_return": bah.total_return,
            "equity_last": float(bah.equity.iloc[-1]),
        },
        "chomo": {
            "total_return": res.total_return,
            "win_rate": res.win_rate,
            "max_drawdown": res.max_drawdown,
            "equity_last": float(res.equity.iloc[-1]),
            "num_trades": len(res.trades),
            "sharpe": sharpe,
            "calmar": calmar,
        },
        "equity_chomo": res.equity,
        "equity_bah": bah.equity,
        "trades": [t.__dict__ for t in res.trades],
    }
