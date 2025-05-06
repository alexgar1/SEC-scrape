13F-Alpha-Miner
A lightweight pipeline that scrapes SEC EDGAR 13-F filings, converts them into structured stock–holding data, augments each record with market fundamentals, and trains ML models that back-test simple “buy” signals.

Quick glance
bash

# 1 / 3  ─ Download recent 13-F filings & holdings tables
python scrape.py

# 2 / 3  ─ Enrich with prices/financials ➜ build X / y datasets
python interpretFilings.py

# 3 / 3  ─ Train & evaluate LR, RF, XGB (+ tiny NN) classifiers
python trainTradeModel.py
All artefacts are written to DATA/.

Project layout
.
├─ scrape.py              # EDGAR crawler → *.feather (13-F filings & holdings)
├─ interpretFilings.py    # Feature builder → X.json / y.json
├─ trainTradeModel.py     # ML training / back-test report
└─ DATA/
   ├─ filings.feather     # Saved by scrape.py        (⇢ TESTfilings.feather if TEST=True)
   ├─ holdings.feather    # Saved by scrape.py
   ├─ X.json              # Training features        (⇢ TESTX.json        if TEST=True)
   ├─ y.json              # Labels / test returns
   └─ tickerCache         # Thread-safe on-disk cache (sqlite-shelve)	


What each stage does
Stage	Key steps
scrape.py	▸ Walks Archives/edgar/data/<CIK> directories without the SEC bulk API
▸ Identifies 13-F “information table” links, parses issuer / shares / value
▸ Saves one row per holding to holdings.feather and minimal metadata to filings.feather
interpretFilings.py	▸ Merges holdings↔filings
▸ Thread-pool per filer to compute: market cap history, volume, EBITDA, debt–cash, % share changes, etc.
▸ Saves nested dicts X.json (features) and y.json (hourly post-filing returns)
trainTradeModel.py	▸ Builds feature matrix & labels
▸ Trains Logistic Regression, Random Forest, XGBoost, and a small Keras NN
▸ Prints classification metrics plus a naïve “buy if prob > 0.5” trading simulation

Road-map / TODO
Extend scraper to 10-K & 8-K (bulk fundamentals & sentiment NLP).
Replace shelve with SQL for concurrent safe caching.
Add walk-forward CV and adaptive position-sizing in back-test.

© 2025 Alex Garrett • MIT License