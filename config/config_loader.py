
from __future__ import annotations
import os, yaml
from dotenv import load_dotenv
from tools.time_utils import to_unix_ms
from backtest.backtester import RunConfig

def load_config_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def build_run_config(cfg_dict: dict) -> RunConfig:
    load_dotenv()
    mongo_uri = os.getenv("MONGODB_URI", cfg_dict.get("mongo_uri", "mongodb://localhost:27017"))
    print(f"MongoDB URI: {mongo_uri}")
    start = cfg_dict.get("start", None)
    end = cfg_dict.get("end", None)
    start_ms = to_unix_ms(start) if start else None
    end_ms = to_unix_ms(end) if end else None
    print(f"start time: {start_ms}, end time: {end_ms}")

    return RunConfig(
        mongo_uri=mongo_uri,
        mongo_db=cfg_dict.get("mongo_db", cfg_dict.get("db", "market")),
        mongo_coll=cfg_dict.get("mongo_coll", cfg_dict.get("coll", "klines")),
        symbol=cfg_dict.get("symbol", "BTCUSDT"),
        interval=cfg_dict.get("interval", "1h"),
        start_ms=start_ms,
        end_ms=end_ms,
        ckpt_path=cfg_dict.get("ckpt_path", cfg_dict.get("ckpt", "")),
        device=cfg_dict.get("device", None),
        fee_bps=float(cfg_dict.get("fee_bps", 1.0)),
        slippage_bps=float(cfg_dict.get("slippage_bps", 0.0)),
        initial_capital=float(cfg_dict.get("initial_capital", cfg_dict.get("initial", 10_000.0))),
        threshold=float(cfg_dict.get("threshold", 0.0)),
        csv_path=cfg_dict.get("csv_path", None),
    )
