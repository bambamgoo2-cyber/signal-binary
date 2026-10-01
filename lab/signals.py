"""Skor sinyal 0-4 untuk NAIK dan TURUN. Sinyal terbit di penutupan bar t."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .config import IndicatorParams

COMPONENTS = ["rsi", "macd", "mom", "frac"]


def _active(event: pd.Series, window: int) -> pd.Series:
    """Kejadian dianggap aktif selama `window` bar terakhir (termasuk bar ini)."""
    return event.astype(float).rolling(max(1, window), min_periods=1).max().astype(bool)


def build_signals(df: pd.DataFrame, p: IndicatorParams) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    out = pd.DataFrame(index=df.index)
    out["rsi"] = ind.rsi(c, p.rsi_period)
    out["roc"] = ind.roc(c, p.roc_period)
    out = out.join(ind.macd(c, p.macd_fast, p.macd_slow, p.macd_signal))
    out["atr"] = ind.atr(h, l, c, p.atr_period)
    out = out.join(ind.fractals(h, l))
    tol = p.fractal_touch_atr * out["atr"]

    # --- 4 komponen, sisi NAIK ---
    rsi_up = (out["rsi"].shift(1) < p.rsi_oversold) & (out["rsi"] >= p.rsi_oversold)
    macd_up = (out["macd"].shift(1) <= out["macd_signal"].shift(1)) & (out["macd"] > out["macd_signal"])
    touch_low = l <= out["last_fractal_low"] + tol
    frac_up = _active(touch_low, p.fractal_lookback) & (c > o) & (c > out["last_fractal_low"])

    # --- sisi TURUN (kebalikan) ---
    rsi_dn = (out["rsi"].shift(1) > p.rsi_overbought) & (out["rsi"] <= p.rsi_overbought)
    macd_dn = (out["macd"].shift(1) >= out["macd_signal"].shift(1)) & (out["macd"] < out["macd_signal"])
    touch_high = h >= out["last_fractal_high"] - tol
    frac_dn = _active(touch_high, p.fractal_lookback) & (c < o) & (c < out["last_fractal_high"])

    out["c_rsi_up"] = _active(rsi_up, p.event_window)
    out["c_macd_up"] = _active(macd_up, p.event_window)
    out["c_mom_up"] = out["roc"] > 0
    out["c_frac_up"] = frac_up
    out["c_rsi_dn"] = _active(rsi_dn, p.event_window)
    out["c_macd_dn"] = _active(macd_dn, p.event_window)
    out["c_mom_dn"] = out["roc"] < 0
    out["c_frac_dn"] = frac_dn

    up_cols = [f"c_{k}_up" for k in COMPONENTS]
    dn_cols = [f"c_{k}_dn" for k in COMPONENTS]
    out["up_score"] = out[up_cols].sum(axis=1).astype(int)
    out["down_score"] = out[dn_cols].sum(axis=1).astype(int)

    up_ok = (out["up_score"] >= p.min_score) & (out["up_score"] > out["down_score"])
    dn_ok = (out["down_score"] >= p.min_score) & (out["down_score"] > out["up_score"])
    out["signal"] = np.where(up_ok, 1, np.where(dn_ok, -1, 0))
    out["score"] = np.where(out["signal"] == 1, out["up_score"], np.where(out["signal"] == -1, out["down_score"], 0))
    return out


def valid_mask(feat: pd.DataFrame) -> np.ndarray:
    """Bar yang indikatornya sudah selesai warm-up."""
    return feat[["rsi", "roc", "macd", "macd_signal", "atr"]].notna().all(axis=1).to_numpy()


def components_text(row: pd.Series, direction: int) -> str:
    suf = "up" if direction > 0 else "dn"
    labels = {"rsi": "RSI", "macd": "MACD", "mom": "MOM", "frac": "FRK"}
    return " ".join(f"{labels[k]}{'✓' if row[f'c_{k}_{suf}'] else '✗'}" for k in COMPONENTS)
