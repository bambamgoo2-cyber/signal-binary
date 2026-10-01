"""Indikator. Setiap nilai pada waktu t hanya memakai data sampai bar t (kausal)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI Wilder."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.where(avg_loss != 0)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return out


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of change (%) = momentum relatif terhadap harga N bar lalu."""
    return close.pct_change(period) * 100.0


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def fractals(high: pd.Series, low: pd.Series) -> pd.DataFrame:
    """Fractal Bill Williams 5 bar (2 bar kiri, 2 bar kanan, pembanding ketat).

    Bar pusat = i. Fractal baru *diketahui* pada bar i+2 (setelah 2 bar konfirmasi).
    Karena itu seluruh logika dipakai lewat shift() sehingga nilai pada baris t
    hanya memakai high/low sampai t. Tidak ada look-ahead.

    Kolom `last_fractal_*` = level fractal terkonfirmasi terakhir (ffill).
    Kolom `display_pivot_*` = posisi pivot pada bar pusatnya, HANYA untuk grafik
    (menggeser data 2 bar ke belakang, jangan dipakai di sinyal).
    """
    h_c, l_c = high.shift(2), low.shift(2)
    is_high = (h_c > high.shift(4)) & (h_c > high.shift(3)) & (h_c > high.shift(1)) & (h_c > high)
    is_low = (l_c < low.shift(4)) & (l_c < low.shift(3)) & (l_c < low.shift(1)) & (l_c < low)

    fh = h_c.where(is_high)
    fl = l_c.where(is_low)
    return pd.DataFrame(
        {
            "last_fractal_high": fh.ffill(),
            "last_fractal_low": fl.ffill(),
            "display_pivot_high": fh.shift(-2),
            "display_pivot_low": fl.shift(-2),
        }
    )
