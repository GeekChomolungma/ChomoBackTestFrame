
from __future__ import annotations
from typing import Dict, Any, Optional
import torch
import numpy as np
import pandas as pd

# Import user's model class from their codebase
from model.network.chomo_transformer import ChomoTransformer_forBTC

def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    return ckpt["model_state"], ckpt.get("meta", {}), ckpt.get("cfg", {})

def rolling_zscore(x: np.ndarray, win: int = 512, eps: float = 1e-8):
    """
    Rolling z-score using ONLY past data.
    x: [T, d] -> out[t] uses stats of x[:t].
    """
    T, d = x.shape
    out = np.empty_like(x, dtype=np.float64)
    buf = np.zeros((win, d), dtype=np.float64)
    csum = np.zeros((d,), dtype=np.float64)       # size [1, d]
    csum2 = np.zeros((d,), dtype=np.float64)      # size [1, d]
    
    # head for rolling update
    head = 0
    # n for window size
    n = 0

    for t in range(T):
        if t == 0: 
            out[t] = 0.0
            continue

        prev = x[t-1].astype(np.float64) # previous vector in t-1, size [1, d]
        if n < win:
            buf[n] = prev; n += 1 # after this plus 1, the n is equal to t.
            # z score calculation depends on [t-w, t-1] mean and std, so just save the prev to buf
            csum += prev; csum2 += prev*prev
        else:
            old = buf[head]; csum -= old; csum2 -= old*old
            buf[head] = prev; csum += prev; csum2 += prev*prev
            head = (head + 1) % win  # a loop mechanism, update the buf one by one

        mean = csum / n
        var = np.maximum(csum2 / n - mean*mean, 0.0)
        std = np.sqrt(var + eps)
        out[t] = ((x[t] - mean) / std).astype(np.float32)
    return out
class ChomoModelWrapper:
    """
    Make sure compatible with training pipeline:
    - features = cfg['data']['x_cols']
    - rolling z-score on x_z_cols with rolling_win
    - window_len = seq_len, this is the data window length for input X
    - predict y_hat(t) with window [t-L+1 ... t] ( matching the train window [t-L+1+rwin ... t+rwin])
    Usage:
        model = ChomoModelWrapper(ckpt_path)
        model.prime_series(df)  # use the entire df for preprocessing
        y_hat = model.predict_logret(window_df)  # use in backtest loop
    """
    def __init__(self, ckpt_path: str, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state_dict, meta, cfg = load_checkpoint(ckpt_path, self.device)
        self.meta = meta
        self.cfg = cfg

        data_cfg = cfg.get("data", {})
        # === key parameters, compatible with loader_general.py ===
        self.horizon    = int(data_cfg.get("horizon",   meta.get("horizon",   1)))
        self.window_len = int(data_cfg.get("seq_len",   meta.get("seq_len",   96)))
        self.rolling_win= int(data_cfg.get("rolling_win", meta.get("rolling_win", 512)))
        self.x_cols     = list(data_cfg.get("x_cols",   []))
        self.x_z_cols   = list(data_cfg.get("x_z_cols", []))
        self.norm       = data_cfg.get("norm", "rolling")

        # rwin, the effective start point offset for rolling z-score
        self.rwin = self.rolling_win if (self.norm == "rolling" and len(self.x_z_cols) > 0) else 0

        # Model(eval process)
        # From train setting
        m = cfg["model"]
        self.model = ChomoTransformer_forBTC(
            d_in=meta["d_in"], d_model=m["d_model"], nhead=m["nhead"],
            num_layers=m["num_layers"], d_ff=m["d_ff"], dropout=m["dropout"],
            out_dim=m["out_dim"], use_last_token=m["use_last_token"]
        ).to(device)
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        # to be fillin prime_series
        self._X_std: Optional[np.ndarray] = None   # Standardized feature matrix [T, d_in]
        self._ts: Optional[np.ndarray] = None      # ts array aligned with X_std [T]
        self._ts_to_idx: Optional[dict] = None     # ts -> row index
        self._d_in: Optional[int] = None

    def _build_features_std(self, df: pd.DataFrame) -> np.ndarray:
        """
        As per training logic, construct X and perform rolling z-score (only for x_z_cols).
        Return
            Standardized feature matrix [T, d_in]
        """
        if not self.x_cols:
            raise ValueError("cfg['data']['x_cols'] is empty. Please provide x_cols in the checkpoint's cfg.")

        X_df = df[self.x_cols].copy().astype(np.float32)

        if self.norm == "rolling" and len(self.x_z_cols) > 0:
            zcols = [c for c in self.x_z_cols if c in X_df.columns]

            print(zcols)
            print(self.x_z_cols)

            # # check if zcols is same as x_z_cols
            # if set(zcols) != set(self.x_z_cols):
            #     raise ValueError("zcols must be the same as x_z_cols")

            arr = X_df[zcols].to_numpy(dtype=np.float32)
            arr = rolling_zscore(arr, win=self.rolling_win)  # should match training
            X_df.loc[:, zcols] = arr

        X = X_df.to_numpy(dtype=np.float32)
        return X

    def prime_series(self, df: pd.DataFrame, ts_col: str = "endtime"):
        """
        Prepare the time series for evaluation.
        - Standardized feature matrix X_std
        - ts mapping
        """
        if ts_col not in df.columns:
            raise ValueError(f"expect df has '{ts_col}' column (ms timestamp)")

        # Ensure chronological order (same as training)
        df_sorted = df.sort_values(ts_col).reset_index(drop=True)
        self._ts = df_sorted[ts_col].to_numpy()
        self._ts_to_idx = {int(t): i for i, t in enumerate(self._ts)}

        X_std = self._build_features_std(df_sorted)  # [T, d_in]
        self._X_std = X_std # this is the X input for the evaluation
        self._d_in = X_std.shape[1]

    def _slice_window_tensor_by_ts(self, this_pred_idx: int) -> torch.Tensor:
        """
        Slice a window tensor [1, L, d_in] ending at t_end_ts.
        """
        if self._X_std is None or self._ts_to_idx is None:
            raise RuntimeError("You must call prime_series(df) once before predictions.")
        
        if this_pred_idx is None:
            raise ValueError(f"this_pred_idx={this_pred_idx} not found in cached series ts.")

        L = self.window_len
        r = self.rwin
        start = this_pred_idx - L
        end   = this_pred_idx
        if start < 0 or end > len(self._X_std):
            # when the window is out of bounds
            raise IndexError(f"window slice out of bounds: start={start}, end={end}, T={len(self._X_std)}")

        xs = self._X_std[start:end, :]  # [L, d_in]
        x = torch.as_tensor(xs, dtype=torch.float32).unsqueeze(0)  # [1, L, d_in]
        return x

    @torch.no_grad()
    def predict_logret(self, this_index: int = 0) -> float:
        """
        Predict the log return for a given end time.
        """

        x = self._slice_window_tensor_by_ts(this_index).to(self.device)  # [1, L, d_in]
        y_hat = self.model(x)  # Assume regression scalar
        return float(y_hat.squeeze().detach().float().cpu().item())

