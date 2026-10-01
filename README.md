# Binary Signal Lab

Alat uji statistik untuk sinyal NAIK/TURUN opsi biner berekspirasi tetap. **Bukan saran investasi.**

## Jalankan
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Uji
```bash
python -m pytest -q        # atau: python tests/test_core.py
```

## Struktur
- `app.py`: UI Streamlit (sidebar parameter, tab Sinyal, Backtest, Metodologi)
- `lab/config.py`: parameter indikator & backtest
- `lab/data.py`: yfinance (1m ≤ ~7 hari; 5m/15m ≤ ~60 hari)
- `lab/indicators.py`: RSI, ROC, MACD, ATR, Fractal 5 bar (terkonfirmasi 2 bar)
- `lab/signals.py`: skor 0-4 dan sinyal
- `lab/backtest.py`: simulasi, metrik, split IS/OOS, pembanding acak
- `lab/charts.py`: grafik Plotly
- `tests/test_core.py`: tes anti look-ahead dan matematika payout
