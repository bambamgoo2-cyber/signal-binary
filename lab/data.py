from __future__ import annotations

import pandas as pd

# Batas data intraday yfinance: 1m maks ~7 hari, 5m/15m maks ~60 hari
PERIOD_OPTIONS = {
    "1m": ["1d", "5d", "7d"],
    "5m": ["5d", "1mo", "60d"],
    "15m": ["5d", "1mo", "60d"],
}
INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15}


def download_ohlc(ticker: str, interval: str, period: str, tz: str = "Asia/Jakarta") -> pd.DataFrame:
    """Unduh OHLC nyata dari yfinance. Kolom: open, high, low, close (huruf kecil)."""
    import yfinance as yf  # impor lambat agar modul inti bisa diuji tanpa yfinance

    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise ValueError(
            f"Tidak ada data untuk {ticker} ({interval}, {period}). "
            "Periksa ticker, atau coba periode lebih pendek."
        )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close"]].copy()
    df.columns = ["open", "high", "low", "close"]
    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")].sort_index()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    try:
        df.index = df.index.tz_convert(tz)
    except Exception:
        df.index = df.index.tz_convert("UTC")
    return df


def drop_unclosed_last_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Bar terakhir dari yfinance sering masih berjalan (belum tutup)."""
    return df.iloc[:-1] if len(df) > 1 else df
