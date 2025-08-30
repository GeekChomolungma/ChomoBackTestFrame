import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

class BacktestPlotter:
    """
    usage:
        out = run_backtest(run_cfg)
        BacktestPlotter().plot(out, title="Chomo Backtest", save_path="backtest.png")
    dependencies:
        - out["klines"]: DataFrame, columns includes ["open","high","low","close","endtime"]
        - out["trades"]: List[dict], each elements: entry_ts, exit_ts, entry_price, exit_price
        - out["equity_chomo"], out["equity_bah"]: pd.Series, index 应与 klines 的 endtime 对齐
        - out["buy_and_hold"]["total_return"]
    """

    def __init__(self, figsize=(14, 9), candle_width=0.6, wick_width=0.8, up_color="#2ca02c", down_color="#d62728"):
        self.figsize = figsize
        self.candle_width = candle_width
        self.wick_width = wick_width
        self.up_color = up_color
        self.down_color = down_color

    def _draw_candles(self, ax, o, h, l, c):
        """
        # Pure matplotlib lightweight K-line drawing: green for rising, red for falling, long bars for bodies, vertical lines for shadows.
        # x-axis: integer index from 0 to n-1.
        """
        n = len(o)
        x = np.arange(n)

        up = c >= o
        down = ~up

        # shadow up/down
        ax.vlines(x, l, h, linewidth=self.wick_width, color=np.where(up, self.up_color, self.down_color))

        # body(with thicker lines)
        body_low = np.minimum(o, c)
        body_high = np.maximum(o, c)
        # use thicker lines to simulate candle bodies
        ax.vlines(x[up], body_low[up], body_high[up], linewidth=self.candle_width*8, color=self.up_color)
        ax.vlines(x[down], body_low[down], body_high[down], linewidth=self.candle_width*8, color=self.down_color)

        ax.set_xlim(-1, n)  # leave some margin

    def _ts_to_index(self, ts_array: np.ndarray, all_ts: np.ndarray) -> Dict[int, int]:
        """
        convert the endtime to index number
        all_ts: numpy array of all timestamps (aligned with klines)

        """
        # build mapping: timestamp -> index
        return {int(t): i for i, t in enumerate(all_ts)}

    def plot(self, out: Dict[str, Any], title: Optional[str] = None, save_path: Optional[str] = None, show: bool = True):
        df: pd.DataFrame = out["klines"]
        # extract required columns
        o = df["open"].to_numpy(dtype=np.float64)
        h = df["high"].to_numpy(dtype=np.float64)
        l = df["low"].to_numpy(dtype=np.float64)
        c = df["close"].to_numpy(dtype=np.float64)
        ts = df["endtime"].to_numpy()  # assume integer timestamps (ms or s)

        # extract equity curves
        eq_chomo: pd.Series = out["equity_chomo"]
        eq_bah: pd.Series = out["equity_bah"]

        # convert the endtime to index number (for marking trade points)
        ts_to_idx = self._ts_to_index(ts, ts)

        # prepare canvas
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.5], hspace=0.07)
        ax_price = fig.add_subplot(gs[0, 0])
        ax_eq = fig.add_subplot(gs[1, 0], sharex=ax_price)

        # === upper plot: K-line + trade markers ===
        self._draw_candles(ax_price, o, h, l, c)

        # Buy & Hold marker
        try:
            bah_start_idx = 0
            bah_end_idx = len(df) - 1
            ax_price.scatter([bah_start_idx], [o[0]], marker='^', s=60, color='tab:blue', label='BAH Buy')
            ax_price.scatter([bah_end_idx], [c[-1]], marker='v', s=60, color='tab:blue', facecolors='none', label='BAH Sell')
        except Exception:
            pass  # error handling

        # Chomo trade markers
        trades = out.get("trades", [])
        chomo_buy_x, chomo_buy_y = [], []
        chomo_sell_x, chomo_sell_y = [], []
        for t in trades:
            ei = ts_to_idx.get(int(t["entry_ts"]))
            xo = ts_to_idx.get(int(t["exit_ts"]))
            if ei is not None:
                chomo_buy_x.append(ei)
                chomo_buy_y.append(t["entry_price"])
            if xo is not None:
                chomo_sell_x.append(xo)
                chomo_sell_y.append(t["exit_price"])

        if len(chomo_buy_x) > 0:
            ax_price.scatter(chomo_buy_x, chomo_buy_y, marker='^', s=50, color='darkorange', label='Chomo Buy', zorder=3)
        if len(chomo_sell_x) > 0:
            ax_price.scatter(chomo_sell_x, chomo_sell_y, marker='v', s=50, color='purple', label='Chomo Sell', zorder=3)

        ax_price.set_ylabel("Price")
        ax_price.grid(True, alpha=0.25)
        ax_price.legend(loc='upper left', ncol=3, frameon=False)

        # x-axis
        x = np.arange(len(df))
        ax_price.set_xticks([])  # upper plot does not show ticks

        # === lower plot: equity curves ===
        # align equity curves by index (map endtime to index)
        # eq_* 's index is endtime; we map it to 0..n-1
        # align array length
        eq_chomo_arr = np.full(len(df), np.nan, dtype=float)
        eq_bah_arr = np.full(len(df), np.nan, dtype=float)

        # write each timestamp position of the Series into the corresponding index
        # assume eq_* 's index is aligned with df["endtime"] (same value set)
        for t_idx, t_val in enumerate(df["endtime"].to_numpy()):
            if t_val in eq_chomo.index:
                eq_chomo_arr[t_idx] = float(eq_chomo.loc[t_val])
            if t_val in eq_bah.index:
                eq_bah_arr[t_idx] = float(eq_bah.loc[t_val])

        ax_eq.plot(x, eq_chomo_arr, label='Chomo Equity', linewidth=1.6)
        ax_eq.plot(x, eq_bah_arr, label='Buy&Hold Equity', linewidth=1.2, alpha=0.9, linestyle='--')
        ax_eq.set_ylabel("Equity")
        ax_eq.set_xlabel("Index (aligned with klines)")  # use index as x-axis as per your requirement
        ax_eq.grid(True, alpha=0.3)
        ax_eq.legend(loc='upper left', frameon=False)

        # title (can display simple metrics)
        if title is None:
            title = "Backtest"
        # include Sharpe/Calmar (if exists)
        sharpe = out.get("chomo", {}).get("sharpe", None)
        calmar = out.get("chomo", {}).get("calmar", None)
        metric_txt = []
        if sharpe is not None:
            metric_txt.append(f"Sharpe: {sharpe:.2f}")
        if calmar is not None:
            metric_txt.append(f"Calmar: {calmar:.2f}")
        if metric_txt:
            title = f"{title}  |  " + "  ·  ".join(metric_txt)

        fig.suptitle(title, y=0.98)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
