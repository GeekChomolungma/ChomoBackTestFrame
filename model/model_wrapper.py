
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
    x: [T, d] -> out[t] uses stats of x[:t] 
    """
    T, d = x.shape
    out = np.empty_like(x, dtype=np.float64)
    buf = np.zeros((win, d), dtype=np.float64)
    csum = np.zeros((d,), dtype=np.float64)
    csum2 = np.zeros((d,), dtype=np.float64)
    head = 0
    n = 0

    for t in range(T):
        if t == 0:
            out[t] = 0.0
            continue

        prev = x[t-1].astype(np.float64)
        if n < win:
            buf[n] = prev; n += 1
            csum += prev; csum2 += prev*prev
        else:
            old = buf[head]; csum -= old; csum2 -= old*old
            buf[head] = prev; csum += prev; csum2 += prev*prev
            head = (head + 1) % win

        mean = csum / n
        var = np.maximum(csum2 / n - mean*mean, 0.0)
        std = np.sqrt(var + eps)
        out[t] = ((x[t] - mean) / std).astype(np.float32)
    return out

class ChomoModelWrapper:
    """
    与训练数据管线保持一致：
    - features = cfg['data']['x_cols']
    - rolling z-score on x_z_cols with rolling_win（历史-only）
    - 窗口长度 = seq_len（注意不是 rolling_win）
    - 预测 y_hat(t) 使用窗口 [t-L+1 ... t]（但在标准化矩阵中索引为 [t-L+1+rwin ... t+rwin]）
    使用方法：
        model = ChomoModelWrapper(ckpt_path)
        model.prime_series(df)  # 先用整段 df 预处理标准化
        y_hat = model.predict_logret(window_df)  # 回测循环里用
    """
    def __init__(self, ckpt_path: str, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state_dict, meta, cfg = load_checkpoint(ckpt_path, self.device)
        self.meta = meta
        self.cfg = cfg

        data_cfg = cfg.get("data", {})
        # === 关键参数，名称对齐你的 loader_general.py ===
        self.horizon    = int(data_cfg.get("horizon",   meta.get("horizon",   1)))
        self.window_len = int(data_cfg.get("seq_len",   meta.get("seq_len",   96)))   # 注意：是 seq_len，不是 rolling_win
        self.rolling_win= int(data_cfg.get("rolling_win", meta.get("rolling_win", 512)))
        self.x_cols     = list(data_cfg.get("x_cols",   []))
        self.x_z_cols   = list(data_cfg.get("x_z_cols", []))
        self.norm       = data_cfg.get("norm", "rolling")

        # rwin 与训练一致：当做“有效起点的偏移”
        self.rwin = self.rolling_win if (self.norm == "rolling" and len(self.x_z_cols) > 0) else 0

        # 模型
        self.model = ChomoTransformer_forBTC(cfg)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        # 将在 prime_series 里填充
        self._X_std: Optional[np.ndarray] = None   # 标准化后的全量特征矩阵 [T, d_in]
        self._ts: Optional[np.ndarray] = None      # 与 X_std 对齐的 ts 数组 [T]
        self._ts_to_idx: Optional[dict] = None     # ts -> 行索引
        self._d_in: Optional[int] = None

    def _build_features_std(self, df: pd.DataFrame) -> np.ndarray:
        """
        按训练逻辑构造 X 并做 rolling z-score（仅对 x_z_cols）。
        返回：标准化后的 X_all，形状 [T, d_in]
        """
        if not self.x_cols:
            raise ValueError("cfg['data']['x_cols'] 为空。需要在 checkpoint 的 cfg 里提供 x_cols。")

        X_df = df[self.x_cols].copy().astype(np.float32)

        if self.norm == "rolling" and len(self.x_z_cols) > 0:
            zcols = [c for c in self.x_z_cols if c in X_df.columns]
            if len(zcols) > 0:
                arr = X_df[zcols].to_numpy(dtype=np.float32)
                arr = rolling_zscore(arr, win=self.rolling_win)  # 与训练完全一致
                X_df.loc[:, zcols] = arr

        X = X_df.to_numpy(dtype=np.float32)
        return X

    def prime_series(self, df: pd.DataFrame, ts_col: str = "ts"):
        """
        必须在回测开始前调用一次。用整段序列生成：
        - 标准化的特征矩阵 X_std
        - ts 映射
        """
        if ts_col not in df.columns:
            raise ValueError(f"expect df has '{ts_col}' column (毫秒时间戳)")

        # 保证时序排序（与训练一致）
        df_sorted = df.sort_values(ts_col).reset_index(drop=True)
        self._ts = df_sorted[ts_col].to_numpy()
        self._ts_to_idx = {int(t): i for i, t in enumerate(self._ts)}

        X_std = self._build_features_std(df_sorted)  # [T, d_in]
        self._X_std = X_std
        self._d_in = X_std.shape[1]

    def _slice_window_tensor_by_ts(self, t_end_ts: int) -> torch.Tensor:
        """
        给定“窗口右端”的时间戳 t_end_ts（即回测循环中的 t 对应 df.iloc[i] 的 ts），
        从已 prime 的 X_std 中切出与训练一致的窗口：
          索引区间 = [i - L + 1 + rwin ... i + rwin]，长度 L
        返回张量 shape [1, L, d_in]
        """
        if self._X_std is None or self._ts_to_idx is None:
            raise RuntimeError("You must call prime_series(df) once before predictions.")

        i = self._ts_to_idx.get(int(t_end_ts), None)
        if i is None:
            raise ValueError(f"t_end_ts={t_end_ts} not found in cached series ts.")

        L = self.window_len
        r = self.rwin
        start = i - L + 1 + r
        end   = i + 1 + r
        if start < 0 or end > len(self._X_std):
            # 头尾不足时无法形成有效窗口（与 __len__ 的裁剪逻辑一致）
            raise IndexError(f"window slice out of bounds: start={start}, end={end}, T={len(self._X_std)}")

        xs = self._X_std[start:end, :]  # [L, d_in]
        x = torch.as_tensor(xs, dtype=torch.float32).unsqueeze(0)  # [1, L, d_in]
        return x

    @torch.no_grad()
    def predict_logret(self, window_df: pd.DataFrame, ts_col: str = "ts") -> float:
        """
        回测里依然可以保持你的调用方式：
            window = df.iloc[i-L+1:i+1]
            y_hat = model_wrapper.predict_logret(window)
        但内部不会用 window_df 做临时标准化，而是用 prime_series 缓存的全局标准化矩阵，
        根据 window_df 的最后一个 ts 来切窗口，保证与训练对齐。
        """
        if ts_col not in window_df.columns:
            raise ValueError(f"window_df must contain '{ts_col}' column")

        t_end_ts = int(window_df[ts_col].iloc[-1])
        x = self._slice_window_tensor_by_ts(t_end_ts).to(self.device)  # [1, L, d_in]
        y_hat = self.model(x)  # 假设回归标量
        return float(y_hat.squeeze().detach().float().cpu().item())

