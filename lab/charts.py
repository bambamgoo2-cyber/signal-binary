from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .backtest import BacktestResult, ST_EXEC, ST_PENDING
from .config import BacktestParams

GREEN, RED, BLUE, GREY, ORANGE = "#26a69a", "#ef5350", "#2979ff", "#9e9e9e", "#ff9800"


def candlestick(df: pd.DataFrame, feat: pd.DataFrame, sim: pd.DataFrame, last_n: int = 300,
                show_fractals: bool = True, title: str = "") -> go.Figure:
    start = max(0, len(df) - last_n)
    d = df.iloc[start:]
    f = feat.iloc[start:]
    x = d.index.strftime("%d-%b %H:%M")  # sumbu kategori: tanpa celah akhir pekan

    fig = go.Figure(go.Candlestick(
        x=x, open=d["open"], high=d["high"], low=d["low"], close=d["close"], name="Harga",
        increasing_line_color=GREEN, decreasing_line_color=RED))

    if show_fractals:
        for col, sym, color, nm in [("display_pivot_high", "diamond", RED, "Fractal high"),
                                    ("display_pivot_low", "diamond", GREEN, "Fractal low")]:
            s = f[col].dropna()
            if len(s):
                fig.add_trace(go.Scatter(x=s.index.strftime("%d-%b %H:%M"), y=s, mode="markers", name=nm,
                                         marker=dict(symbol=sym, size=5, color=color, opacity=0.6)))

    if not sim.empty:
        s = sim[sim["entry_idx"] >= start]
        off = (d["high"] - d["low"]).rolling(20, min_periods=1).mean().to_numpy()
        for direction, sym, color in [("NAIK", "triangle-up", GREEN), ("TURUN", "triangle-down", RED)]:
            for executed, opacity, label in [(True, 1.0, "dieksekusi"), (False, 0.45, "dilewati")]:
                m = s["status"].isin([ST_EXEC, ST_PENDING]) if executed else ~s["status"].isin([ST_EXEC, ST_PENDING])
                g = s[(s["direction"] == direction) & m]
                if g.empty:
                    continue
                pos = g["entry_idx"].to_numpy() - start
                y = (d["low"].to_numpy()[pos] - off[pos]) if direction == "NAIK" else (d["high"].to_numpy()[pos] + off[pos])
                txt = [f"{r.direction} skor {r.score}<br>{r.status}<br>{r.result or '-'}" for r in g.itertuples()]
                fig.add_trace(go.Scatter(
                    x=d.index.strftime("%d-%b %H:%M")[pos], y=y, mode="markers", text=txt, hoverinfo="text+x",
                    name=f"{direction} ({label})",
                    marker=dict(symbol=sym, size=12, color=color, opacity=opacity,
                                line=dict(width=1, color="white"))))

    fig.update_layout(title=title, height=620, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=1.02, x=0), xaxis=dict(type="category", nticks=12, tickangle=-45))
    return fig


def equity_chart(res: BacktestResult, bp: BacktestParams) -> go.Figure:
    pnl = res.used["pnl"].to_numpy(float) if not res.used.empty else np.array([])
    eq = np.concatenate([[bp.initial_capital], bp.initial_capital + np.cumsum(pnl)])
    fig = go.Figure()
    b = res.equity_band
    if not b.empty:
        fig.add_trace(go.Scatter(x=b["trade_no"], y=b["p95"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=b["trade_no"], y=b["p05"], fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(158,158,158,0.25)", name="Acak: rentang 5-95%"))
        fig.add_trace(go.Scatter(x=b["trade_no"], y=b["p50"], line=dict(color=GREY, dash="dot"), name="Acak: median"))
    fig.add_trace(go.Scatter(x=np.arange(len(eq)), y=eq, line=dict(color=BLUE, width=2.5), name="Strategi"))
    fig.add_hline(y=bp.initial_capital, line_dash="dash", line_color=GREY, opacity=0.5)
    n_is = res.metrics["IS"].n
    if 0 < n_is < len(eq):
        fig.add_vline(x=n_is, line_dash="dash", line_color=ORANGE,
                      annotation_text="in-sample | out-of-sample", annotation_position="top")
    fig.update_layout(height=420, xaxis_title="Nomor trade", yaxis_title="Ekuitas", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.1, x=0))
    return fig


def winrate_hist(base, strat_wr: float, breakeven: float, title: str) -> go.Figure:
    fig = go.Figure()
    if len(base.wr_samples):
        fig.add_trace(go.Histogram(x=base.wr_samples * 100, marker_color=GREY, name="Acak", nbinsx=30))
    if not np.isnan(strat_wr):
        fig.add_vline(x=strat_wr * 100, line_color=BLUE, line_width=3, annotation_text="Strategi")
    fig.add_vline(x=breakeven * 100, line_color=ORANGE, line_dash="dash", annotation_text="Break-even",
                  annotation_position="bottom right")
    fig.update_layout(title=title, height=300, xaxis_title="Win rate (%)", yaxis_title="Frekuensi",
                      showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    return fig
