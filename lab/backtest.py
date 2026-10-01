"""Simulasi backtest opsi biner ekspirasi tetap + metrik + pembanding acak."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import BacktestParams

ST_EXEC = "DIEKSEKUSI"
ST_PENDING = "MENUNGGU"
ST_OVERLAP = "DILEWATI (posisi terbuka)"
ST_LIMIT = "DILEWATI (batas rugi harian)"


# ----------------------------------------------------------------------------- simulasi
def simulate(df: pd.DataFrame, feat: pd.DataFrame, bp: BacktestParams) -> pd.DataFrame:
    """Jalankan semua sinyal secara berurutan.

    Entri di harga close bar sinyal; hasil dinilai di close bar t+N.
    Menang +stake*payout, kalah -stake, seri = refund (0).
    Batas rugi harian hanya memakai trade yang SUDAH selesai (exit <= waktu sinyal).
    """
    close = df["close"].to_numpy(float)
    n, N = len(close), bp.expiry_bars
    days = np.array(df.index.date)
    sig = feat["signal"].to_numpy()
    score = feat["score"].to_numpy()

    rows, last_exit, day_book = [], -1, {}
    for i in np.flatnonzero(sig != 0):
        d, exit_i, day = int(sig[i]), int(i) + N, days[i]
        status, result, pnl, exit_price = ST_EXEC, "", np.nan, np.nan

        if (not bp.allow_overlap) and i < last_exit:
            status = ST_OVERLAP
        elif bp.daily_loss_limit > 0:
            settled = sum(p for e, p in day_book.get(day, []) if e <= i)
            if settled <= -bp.daily_loss_limit:
                status = ST_LIMIT

        if status == ST_EXEC:
            last_exit = exit_i
            if exit_i >= n:
                status = ST_PENDING
            else:
                exit_price = close[exit_i]
                diff = (exit_price - close[i]) * d
                eps = 1e-9 * abs(close[i])
                if diff > eps:
                    result, pnl = "MENANG", bp.stake * bp.payout
                elif diff < -eps:
                    result, pnl = "KALAH", -bp.stake
                else:
                    result, pnl = "SERI", 0.0
                day_book.setdefault(day, []).append((exit_i, pnl))

        rows.append(
            {
                "entry_idx": int(i),
                "exit_idx": exit_i if exit_i < n else np.nan,
                "time": df.index[i],
                "exit_time": df.index[exit_i] if exit_i < n else pd.NaT,
                "direction": "NAIK" if d > 0 else "TURUN",
                "score": int(score[i]),
                "entry_price": close[i],
                "exit_price": exit_price,
                "status": status,
                "result": result,
                "pnl": pnl,
            }
        )
    cols = ["entry_idx", "exit_idx", "time", "exit_time", "direction", "score",
            "entry_price", "exit_price", "status", "result", "pnl"]
    return pd.DataFrame(rows, columns=cols)


def label_segments(sim: pd.DataFrame, split_idx: int) -> pd.DataFrame:
    """IS: entri & exit sebelum split. OOS: entri di/ setelah split.
    Trade yang melintasi batas split dibuang (PURGE) agar tidak bocor."""
    sim = sim.copy()
    sim["segment"] = np.where(sim["entry_idx"] < split_idx, "IS", "OOS")
    straddle = (sim["result"] != "") & (sim["entry_idx"] < split_idx) & (sim["exit_idx"] >= split_idx)
    sim.loc[straddle, "segment"] = "PURGE"
    return sim


# ----------------------------------------------------------------------------- statistik
def breakeven_winrate(payout: float) -> float:
    """Win rate impas (seri tidak dihitung): w*payout = (1-w)  ->  w = 1/(1+payout)."""
    return 1.0 / (1.0 + payout)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) untuk X ~ Binomial(n, p). Uji satu sisi: apakah win rate > break-even?"""
    if n == 0:
        return float("nan")
    if k <= 0:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    lg = math.lgamma
    terms = [lg(n + 1) - lg(j + 1) - lg(n - j + 1) + j * lp + (n - j) * lq for j in range(k, n + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


@dataclass
class Metrics:
    n: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    win_rate: float = float("nan")
    breakeven: float = float("nan")
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    p_value: float = float("nan")
    ev_per_trade: float = float("nan")
    ev_pct_stake: float = float("nan")
    total_pnl: float = 0.0
    max_dd: float = 0.0
    max_dd_pct: float = 0.0
    final_equity: float = float("nan")


def equity_curve(pnl: np.ndarray, capital: float) -> np.ndarray:
    return capital + np.cumsum(pnl)


def max_drawdown(pnl: np.ndarray, capital: float) -> tuple[float, float]:
    if len(pnl) == 0:
        return 0.0, 0.0
    eq = equity_curve(pnl, capital)
    peak = np.maximum.accumulate(np.concatenate([[capital], eq]))[1:]
    dd = peak - eq
    return float(dd.max()), float((dd / peak).max())


def compute_metrics(trades: pd.DataFrame, bp: BacktestParams) -> Metrics:
    m = Metrics(breakeven=breakeven_winrate(bp.payout))
    if trades.empty:
        return m
    pnl = trades["pnl"].to_numpy(float)
    m.n = len(trades)
    m.wins = int((trades["result"] == "MENANG").sum())
    m.losses = int((trades["result"] == "KALAH").sum())
    m.ties = int((trades["result"] == "SERI").sum())
    decided = m.wins + m.losses
    if decided:
        m.win_rate = m.wins / decided
        m.ci_low, m.ci_high = wilson_ci(m.wins, decided)
        m.p_value = binom_sf(m.wins, decided, m.breakeven)
    m.ev_per_trade = float(pnl.mean())
    m.ev_pct_stake = m.ev_per_trade / bp.stake * 100.0
    m.total_pnl = float(pnl.sum())
    m.max_dd, m.max_dd_pct = max_drawdown(pnl, bp.initial_capital)
    m.final_equity = bp.initial_capital + m.total_pnl
    return m


# ----------------------------------------------------------------------------- pembanding acak
@dataclass
class Baseline:
    wr_samples: np.ndarray = field(default_factory=lambda: np.array([]))
    ev_samples: np.ndarray = field(default_factory=lambda: np.array([]))
    wr_mean: float = float("nan")
    wr_p05: float = float("nan")
    wr_p95: float = float("nan")
    ev_mean: float = float("nan")
    p_empirical: float = float("nan")  # P(win rate acak >= win rate strategi)


def _outcomes_to_pnl(o: np.ndarray, bp: BacktestParams) -> np.ndarray:
    return np.where(o > 0, bp.stake * bp.payout, np.where(o < 0, -bp.stake, 0.0))


def random_outcomes(close: np.ndarray, valid: np.ndarray, lo: int, hi: int, k: int,
                    bp: BacktestParams, rng: np.random.Generator) -> np.ndarray:
    """Matriks (sims x k) hasil +1/0/-1 untuk entri acak di bar [lo, hi) dengan arah acak."""
    N, n, M = bp.expiry_bars, len(close), bp.n_random_sims
    hi = min(hi, n - N)
    cand = np.flatnonzero(valid[max(lo, 0):max(hi, 0)]) + max(lo, 0)
    if k == 0 or len(cand) == 0:
        return np.zeros((M, 0))
    diff = close[cand + N] - close[cand]
    eps = 1e-9 * np.abs(close[cand])
    sgn = np.where(np.abs(diff) <= eps, 0, np.sign(diff))
    replace = len(cand) < k
    pick = np.empty((M, k), dtype=np.int64)
    for m in range(M):
        pick[m] = np.sort(rng.choice(len(cand), size=k, replace=replace))
    dirs = rng.integers(0, 2, size=(M, k)) * 2 - 1
    return dirs * sgn[pick]


def summarize_baseline(o: np.ndarray, strat: Metrics, bp: BacktestParams) -> Baseline:
    if o.shape[1] == 0:
        return Baseline()
    wins, losses = (o > 0).sum(1), (o < 0).sum(1)
    wr = wins / np.maximum(wins + losses, 1)
    ev = _outcomes_to_pnl(o, bp).mean(1)
    p_emp = float("nan")
    if not math.isnan(strat.win_rate):
        p_emp = float((np.sum(wr >= strat.win_rate) + 1) / (len(wr) + 1))
    return Baseline(wr, ev, float(wr.mean()), float(np.percentile(wr, 5)),
                    float(np.percentile(wr, 95)), float(ev.mean()), p_emp)


# ----------------------------------------------------------------------------- pipeline
@dataclass
class BacktestResult:
    sim: pd.DataFrame                 # semua sinyal + status + segmen
    used: pd.DataFrame                # trade selesai & dieksekusi di IS+OOS (urut waktu)
    metrics: dict                     # "IS", "OOS", "ALL" -> Metrics
    baselines: dict                   # "IS", "OOS" -> Baseline
    equity_band: pd.DataFrame         # kolom: trade_no, p05, p50, p95 (kurva ekuitas acak)
    split_idx: int
    split_time: pd.Timestamp
    n_purged: int


def run_backtest(df: pd.DataFrame, feat: pd.DataFrame, valid: np.ndarray, bp: BacktestParams) -> BacktestResult:
    n, N = len(df), bp.expiry_bars
    split_idx = int(n * (1.0 - bp.oos_fraction))
    sim = label_segments(simulate(df, feat, bp), split_idx)

    done = sim[sim["result"] != ""]
    is_tr = done[done["segment"] == "IS"]
    oos_tr = done[done["segment"] == "OOS"]
    used = pd.concat([is_tr, oos_tr]).sort_values("entry_idx").reset_index(drop=True)

    metrics = {"IS": compute_metrics(is_tr, bp), "OOS": compute_metrics(oos_tr, bp), "ALL": compute_metrics(used, bp)}

    close = df["close"].to_numpy(float)
    rng = np.random.default_rng(bp.seed)
    o_is = random_outcomes(close, valid, 0, split_idx - N, metrics["IS"].n, bp, rng)
    o_oos = random_outcomes(close, valid, split_idx, n - N, metrics["OOS"].n, bp, rng)
    baselines = {"IS": summarize_baseline(o_is, metrics["IS"], bp),
                 "OOS": summarize_baseline(o_oos, metrics["OOS"], bp)}

    o_all = np.hstack([o_is, o_oos])
    if o_all.shape[1] > 0:
        eq = bp.initial_capital + np.cumsum(_outcomes_to_pnl(o_all, bp), axis=1)
        band = pd.DataFrame({"trade_no": np.arange(1, eq.shape[1] + 1),
                             "p05": np.percentile(eq, 5, axis=0),
                             "p50": np.percentile(eq, 50, axis=0),
                             "p95": np.percentile(eq, 95, axis=0)})
    else:
        band = pd.DataFrame(columns=["trade_no", "p05", "p50", "p95"])

    return BacktestResult(sim, used, metrics, baselines, band, split_idx,
                          df.index[min(split_idx, n - 1)], int((sim["segment"] == "PURGE").sum()))
