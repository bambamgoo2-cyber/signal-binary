from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IndicatorParams:
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    roc_period: int = 10
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    # Pantulan fractal: harga menyentuh level fractal (toleransi = x ATR) dalam k bar terakhir
    fractal_touch_atr: float = 0.5
    fractal_lookback: int = 3
    # Kejadian RSI-keluar-zona dan MACD-cross dianggap "aktif" selama k bar (1 = bar yang sama persis)
    event_window: int = 3
    min_score: int = 3


@dataclass(frozen=True)
class BacktestParams:
    expiry_bars: int = 3
    payout: float = 0.80          # 0.80 = untung 80% dari stake bila menang
    stake: float = 10.0
    initial_capital: float = 1000.0
    daily_loss_limit: float = 30.0  # dalam satuan uang; 0 = nonaktif
    allow_overlap: bool = False     # False = satu posisi terbuka pada satu waktu
    oos_fraction: float = 0.30
    n_random_sims: int = 500
    seed: int = 42
