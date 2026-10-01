"""Binary Signal Lab: alat uji statistik sinyal NAIK/TURUN (bukan saran investasi).
Jalankan:  streamlit run app.py
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from lab import charts
from lab.backtest import ST_EXEC, ST_LIMIT, ST_PENDING, breakeven_winrate, run_backtest
from lab.config import BacktestParams, IndicatorParams
from lab.data import INTERVAL_MINUTES, PERIOD_OPTIONS, download_ohlc, drop_unclosed_last_bar
from lab.signals import build_signals, components_text, valid_mask

st.set_page_config(page_title="Binary Signal Lab", layout="wide")

PRESETS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "BTC-USD", "ETH-USD", "GC=F", "(ketik sendiri)"]
DISCLAIMER = (
    "**Bukan saran investasi.** Aplikasi ini hanya alat uji statistik atas data historis. Hasil backtest tidak "
    "menjamin hasil masa depan. Opsi biner berisiko sangat tinggi dan di banyak yurisdiksi (termasuk Indonesia, "
    "menurut Bappebti) tidak diizinkan; periksa regulasi setempat sebelum memakai uang sungguhan."
)


@st.cache_data(ttl=60, show_spinner="Mengunduh data dari yfinance...")
def load(ticker: str, interval: str, period: str, tz: str) -> pd.DataFrame:
    return download_ohlc(ticker, interval, period, tz)


def fmt_pct(x, d=1):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{d}f}%"


def fmt_num(x, d=2):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.{d}f}"


# ------------------------------------------------------------------ sidebar
st.title("Binary Signal Lab")
st.warning(DISCLAIMER, icon="⚠️")

with st.sidebar.form("params"):
    st.header("Data")
    preset = st.selectbox("Aset", PRESETS)
    custom = st.text_input("Ticker manual", "EURUSD=X") if preset == "(ketik sendiri)" else ""
    ticker = (custom or "EURUSD=X").strip() if preset == "(ketik sendiri)" else preset
    interval = st.selectbox("Timeframe", list(PERIOD_OPTIONS), index=1)
    period = st.selectbox("Rentang data", PERIOD_OPTIONS[interval], index=len(PERIOD_OPTIONS[interval]) - 1,
                          help="Batas yfinance: 1m maks ~7 hari, 5m/15m maks ~60 hari.")
    tz = st.text_input("Zona waktu", "Asia/Jakarta", help="Juga menentukan batas 'hari' untuk batas rugi harian.")
    drop_last = st.checkbox("Buang bar terakhir (belum tutup)", value=True)

    st.header("Indikator")
    rsi_p = st.number_input("RSI periode", 2, 100, 14)
    c1, c2 = st.columns(2)
    rsi_os = c1.number_input("Oversold", 1.0, 49.0, 30.0)
    rsi_ob = c2.number_input("Overbought", 51.0, 99.0, 70.0)
    roc_p = st.number_input("Momentum/ROC periode", 1, 100, 10)
    m1, m2, m3 = st.columns(3)
    mf = m1.number_input("MACD cepat", 2, 100, 12)
    ms = m2.number_input("lambat", 3, 200, 26)
    mg = m3.number_input("sinyal", 2, 100, 9)
    frac_tol = st.number_input("Toleransi sentuh fractal (x ATR)", 0.0, 3.0, 0.5, 0.1)
    frac_lb = st.number_input("Cek pantulan dalam k bar terakhir", 1, 10, 3)
    ev_win = st.number_input("Jendela kejadian RSI/MACD (bar)", 1, 10, 3,
                             help="1 = RSI keluar zona & MACD cross harus persis di bar yang sama.")
    min_score = st.slider("Skor minimum sinyal", 1, 4, 3)

    st.header("Backtest")
    expiry = st.selectbox("Ekspirasi (bar ke depan)", [1, 3, 5], index=1)
    payout_pct = st.number_input("Payout (%)", 1.0, 200.0, 80.0, 1.0)
    stake = st.number_input("Stake per trade", 0.01, 1e9, 10.0)
    capital = st.number_input("Modal awal", 1.0, 1e12, 1000.0)
    dll = st.number_input("Batas rugi harian (0 = nonaktif)", 0.0, 1e12, 30.0,
                          help="Setelah rugi bersih hari itu (trade yang sudah selesai) mencapai batas, sinyal baru dilewati.")
    overlap = st.checkbox("Izinkan posisi tumpang-tindih", value=False)
    oos_frac = st.slider("Porsi out-of-sample", 0.10, 0.50, 0.30, 0.05)
    sims = st.number_input("Simulasi acak", 100, 5000, 500, 100)
    seed = st.number_input("Seed acak", 0, 10**6, 42)
    chart_n = st.number_input("Jumlah bar di grafik", 50, 2000, 300, 50)
    st.form_submit_button("Jalankan analisis", type="primary", use_container_width=True)

ip = IndicatorParams(int(rsi_p), float(rsi_os), float(rsi_ob), int(roc_p), int(mf), int(ms), int(mg),
                     14, float(frac_tol), int(frac_lb), int(ev_win), int(min_score))
bp = BacktestParams(int(expiry), payout_pct / 100.0, float(stake), float(capital), float(dll), bool(overlap),
                    float(oos_frac), int(sims), int(seed))

if ip.macd_fast >= ip.macd_slow:
    st.error("MACD: periode cepat harus lebih kecil dari periode lambat.")
    st.stop()

# ------------------------------------------------------------------ data & perhitungan
try:
    df = load(ticker, interval, period, tz)
except Exception as e:  # noqa: BLE001
    st.error(f"Gagal mengambil data: {e}")
    st.stop()

if drop_last:
    df = drop_unclosed_last_bar(df)
if len(df) < 200:
    st.error(f"Data terlalu sedikit ({len(df)} bar). Perpanjang rentang data atau ganti timeframe.")
    st.stop()

feat = build_signals(df, ip)
valid = valid_mask(feat)
res = run_backtest(df, feat, valid, bp)
sim = res.sim

tab_sig, tab_bt, tab_info = st.tabs(["Sinyal & Grafik", "Backtest", "Metodologi"])

# ------------------------------------------------------------------ TAB 1
with tab_sig:
    last = feat.iloc[-1]
    last_time = df.index[-1]
    step = pd.Timedelta(minutes=INTERVAL_MINUTES[interval])
    direction = int(last["signal"])
    last_row = sim[sim["entry_idx"] == len(df) - 1]
    last_status = last_row["status"].iloc[0] if not last_row.empty else ""

    a, b, c, d = st.columns(4)
    a.metric("Bar terakhir (tutup)", last_time.strftime("%d-%b %H:%M"))
    b.metric("Harga", f"{df['close'].iloc[-1]:.5g}")
    c.metric("Sinyal bar ini", {1: "NAIK", -1: "TURUN", 0: "TIDAK ADA"}[direction],
             f"skor {int(last['score'])}" if direction else f"naik {int(last['up_score'])} / turun {int(last['down_score'])}")
    d.metric("Kedaluwarsa jika masuk", (last_time + step * bp.expiry_bars).strftime("%H:%M") if direction else "-")

    today = df.index[-1].date()
    today_done = sim[(sim["result"] != "") & (sim["time"].dt.date == today)]
    today_pnl = float(today_done["pnl"].sum()) if not today_done.empty else 0.0
    if bp.daily_loss_limit > 0 and today_pnl <= -bp.daily_loss_limit:
        st.error(f"Batas rugi harian tercapai (PnL simulasi hari ini {today_pnl:,.2f}). Sinyal baru dilewati.")
    else:
        st.caption(f"PnL simulasi hari ini: {today_pnl:,.2f} | batas rugi harian: "
                   f"{'nonaktif' if bp.daily_loss_limit <= 0 else f'{bp.daily_loss_limit:,.2f}'}")
    if direction and last_status in (ST_LIMIT,):
        st.error("Sinyal terbit, tetapi diblokir oleh batas rugi harian / posisi terbuka.")
    st.caption("Sinyal dihitung dari bar yang sudah tutup; entri disimulasikan di harga close bar sinyal.")

    st.subheader("Sinyal terbaru")
    if sim.empty:
        st.info("Belum ada sinyal pada data ini dengan parameter saat ini. Coba turunkan skor minimum atau perbesar jendela kejadian.")
    else:
        t = sim.sort_values("entry_idx", ascending=False).head(25).copy()
        t["Komponen"] = [components_text(feat.iloc[i], 1 if dr == "NAIK" else -1)
                         for i, dr in zip(t["entry_idx"], t["direction"])]
        t["Kedaluwarsa"] = t["time"] + step * bp.expiry_bars
        show = t.rename(columns={"time": "Waktu", "direction": "Arah", "score": "Skor", "entry_price": "Harga entri",
                                 "exit_price": "Harga exit", "status": "Status", "result": "Hasil", "pnl": "PnL"})
        show["Waktu"] = show["Waktu"].dt.strftime("%d-%b %H:%M")
        show["Kedaluwarsa"] = show["Kedaluwarsa"].dt.strftime("%d-%b %H:%M")
        st.dataframe(show[["Waktu", "Arah", "Skor", "Komponen", "Harga entri", "Kedaluwarsa", "Harga exit",
                           "Status", "Hasil", "PnL"]], hide_index=True, use_container_width=True)

    show_fr = st.checkbox("Tampilkan pivot fractal (hanya visual)", value=True)
    st.plotly_chart(charts.candlestick(df, feat, sim, int(chart_n), show_fr, f"{ticker} | {interval}"),
                    use_container_width=True)
    st.caption("Segitiga terisi = trade dieksekusi/menunggu; segitiga pudar = dilewati (posisi terbuka atau batas rugi harian).")

# ------------------------------------------------------------------ TAB 2
with tab_bt:
    be = breakeven_winrate(bp.payout)
    st.markdown(f"**Break-even win rate = 1 / (1 + payout) = {be * 100:.2f}%** (payout {payout_pct:.0f}%, seri = refund). "
                f"Pemisahan: {len(df):,} bar, out-of-sample mulai **{res.split_time.strftime('%d-%b %H:%M')}** "
                f"({res.n_purged} trade melintasi batas dibuang/purge).")

    def verdict(name: str, m):
        decided = m.wins + m.losses
        if decided < 30:
            st.info(f"{name}: hanya {decided} trade menang/kalah, terlalu sedikit untuk kesimpulan statistik (minimal ~30, idealnya ratusan).")
        elif m.win_rate < m.breakeven:
            st.warning(f"{name}: win rate {m.win_rate * 100:.1f}% DI BAWAH break-even {m.breakeven * 100:.1f}%. "
                       f"Expected value negatif ({m.ev_pct_stake:.1f}% dari stake per trade).", icon="🚨")
        elif m.p_value >= 0.05:
            st.warning(f"{name}: win rate {m.win_rate * 100:.1f}% di atas break-even, tetapi TIDAK signifikan "
                       f"secara statistik (p = {m.p_value:.3f}). Bisa saja kebetulan.")
        else:
            st.success(f"{name}: win rate {m.win_rate * 100:.1f}% > break-even dan p = {m.p_value:.3f}. "
                       "Lolos uji sederhana, tetapi belum bukti edge (lihat risiko overfitting di tab Metodologi).")

    verdict("In-sample", res.metrics["IS"])
    verdict("Out-of-sample", res.metrics["OOS"])

    def col(m):
        return {
            "Jumlah trade": f"{m.n}",
            "Menang / Kalah / Seri": f"{m.wins} / {m.losses} / {m.ties}",
            "Win rate": fmt_pct(m.win_rate),
            "CI 95% (Wilson)": f"{fmt_pct(m.ci_low)} - {fmt_pct(m.ci_high)}",
            "Break-even": fmt_pct(m.breakeven, 2),
            "p-value (win rate > break-even)": fmt_num(m.p_value, 4),
            "EV per trade": fmt_num(m.ev_per_trade),
            "EV (% stake)": f"{fmt_num(m.ev_pct_stake)}%",
            "Total PnL": fmt_num(m.total_pnl),
            "Max drawdown": f"{fmt_num(m.max_dd)} ({fmt_pct(m.max_dd_pct)})",
            "Ekuitas akhir": fmt_num(m.final_equity),
        }

    st.subheader("Hasil strategi")
    st.dataframe(pd.DataFrame({"In-sample": col(res.metrics["IS"]), "Out-of-sample": col(res.metrics["OOS"]),
                               "IS + OOS": col(res.metrics["ALL"])}), use_container_width=True)

    st.subheader("Pembanding acak (entri & arah acak, jumlah trade sama)")
    rows = {}
    for k, nm in [("IS", "In-sample"), ("OOS", "Out-of-sample")]:
        bsl = res.baselines[k]
        rows[nm] = {
            "Win rate acak (rata-rata)": fmt_pct(bsl.wr_mean),
            "Rentang 5-95%": f"{fmt_pct(bsl.wr_p05)} - {fmt_pct(bsl.wr_p95)}",
            "EV acak per trade": fmt_num(bsl.ev_mean),
            "p-empiris (acak >= strategi)": fmt_num(bsl.p_empirical, 3),
        }
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption("Perhatikan: acak pun rugi karena payout < 100%. Strategi harus mengalahkan acak DAN melewati break-even.")

    st.subheader("Kurva ekuitas")
    st.plotly_chart(charts.equity_chart(res, bp), use_container_width=True)

    h1, h2 = st.columns(2)
    h1.plotly_chart(charts.winrate_hist(res.baselines["IS"], res.metrics["IS"].win_rate, be, "Win rate acak: in-sample"),
                    use_container_width=True)
    h2.plotly_chart(charts.winrate_hist(res.baselines["OOS"], res.metrics["OOS"].win_rate, be, "Win rate acak: out-of-sample"),
                    use_container_width=True)

    with st.expander("Daftar semua sinyal / trade"):
        out = sim.copy()
        st.dataframe(out, hide_index=True, use_container_width=True)
        st.download_button("Unduh CSV", out.to_csv(index=False).encode(), f"{ticker}_{interval}_trades.csv", "text/csv")

# ------------------------------------------------------------------ TAB 3
with tab_info:
    st.markdown(f"""
**Sinyal (skor 0-4, terbit jika skor ≥ {ip.min_score} dan lebih besar dari skor sisi lawan)**

| Komponen | NAIK | TURUN |
|---|---|---|
| RSI | keluar dari oversold (< {ip.rsi_oversold:g} lalu ≥ {ip.rsi_oversold:g}) | keluar dari overbought |
| MACD | garis MACD memotong signal ke atas | memotong ke bawah |
| Momentum | ROC({ip.roc_period}) > 0 | ROC < 0 |
| Fractal | harga menyentuh fractal low terkonfirmasi (±{ip.fractal_touch_atr:g} ATR) dalam {ip.fractal_lookback} bar & close > open | kebalikannya di fractal high |

Kejadian RSI dan MACD dianggap aktif selama {ip.event_window} bar terakhir (1 = bar yang sama persis).

**Anti look-ahead.** Fractal 5 bar baru dipakai 2 bar setelah bar pusatnya (shift(2)). Semua indikator kausal;
tes otomatis `tests/test_core.py` memotong data lalu memastikan sinyal historis tidak berubah.

**Backtest.** Entri di close bar sinyal, penilaian di close bar t+{bp.expiry_bars}: menang +stake×payout, kalah −stake, seri 0.
Satu posisi sekaligus (kecuali diizinkan tumpang-tindih). Batas rugi harian hanya memakai trade yang sudah selesai.
Trade yang melintasi batas in-sample/out-of-sample dibuang.

**Cara membaca.** Win rate harus melewati break-even, mengalahkan acak, dan bertahan di out-of-sample.
Jika Anda menyetel parameter sambil melihat hasil out-of-sample, data itu sudah tidak "bersih" lagi.
Mencoba banyak kombinasi parameter juga menaikkan peluang lolos karena kebetulan (multiple testing).

**Keterbatasan.** Data yfinance tidak mencakup spread/harga broker; entri di close bar sinyal itu idealisasi
(praktiknya ada keterlambatan); pembanding acak tidak menerapkan aturan tumpang-tindih/batas harian.

{DISCLAIMER}
""")
