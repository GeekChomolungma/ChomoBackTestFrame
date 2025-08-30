
import argparse, json, os
import pandas as pd
from config.config_loader import load_config_yaml, build_run_config
from backtest.backtester import run_backtest
from bt_plotter.backtest_plotter import BacktestPlotter

def parse_args():
    ap = argparse.ArgumentParser(description="Run backtest with YAML config + .env")
    ap.add_argument("--config", "-c", default="config.yaml", help="Path to config YAML")
    ap.add_argument("--out", default="bt_out", help="Output directory")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg_dict = load_config_yaml(args.config)
    run_cfg = build_run_config(cfg_dict)
    out = run_backtest(run_cfg)
    BacktestPlotter().plot(out, title="Chomo Backtest", save_path=None, show=True)
    
    summary = {
        "buy_and_hold": out["buy_and_hold"],
        "chomo": {k: out["chomo"][k] for k in ["total_return","win_rate","max_drawdown","equity_last","num_trades","sharpe","calmar"]},
    }
    print(json.dumps(summary, indent=2))

    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(out["trades"]).to_csv(os.path.join(args.out, "trades.csv"), index=False)
    out["equity_chomo"].to_csv(os.path.join(args.out, "equity_chomo.csv"), header=True)
    out["equity_bah"].to_csv(os.path.join(args.out, "equity_bah.csv"), header=True)
    print(f"Saved details to {args.out}/")

if __name__ == "__main__":
    main()
