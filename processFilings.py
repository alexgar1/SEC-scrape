# Written by Alex Garrett alexgarrett2468 [ at ] gmail.com
# This script processes SEC filings and extracts features for training/testing a model.
# It circumvents the rate limits of the Yahoo Finance API by caching ticker data in a DuckDB database.
# It uses multithreading to process multiple filers concurrently, extracting stock information and calculating features for each holding.


# TODO:
# get beta
# ensure dates are in UTC


import os
import time
import json
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import duckdb
import requests
import pandas_market_calendars as mcal
from tqdm import tqdm
import logging
import yfinance as yf
from io import StringIO
from difflib import get_close_matches
from pathlib import Path
from functools import lru_cache
import random
import re
import urllib.parse
from contextvars import ContextVar
from time import sleep
import threading




# --- Configuration ---
SOURCE_DB = 'DATA/sec_filings.duckdb'      # existing filings+holdings
FEATURE_DB = 'DATA/sec_features.duckdb'    # new DB for train/test
MAX_WORKERS = min(os.cpu_count() or 1, 8)
BACKPERIOD = 10 # DAYS to look back and forward for stock data
TESTPERIOD = 3 # DAYS post annoucement to assess a price movement (7 hours in normal market day).


@lru_cache(maxsize=1)
def _lazy_load(mapping_file: str | Path) -> dict:
    with open(mapping_file, "r") as f:
        return json.load(f)
    
TICKERS = Path('DATA/companyCIK.json')
TICKERMAPPING = _lazy_load(TICKERS)

HEADERS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ... Chrome/58.0.3029.110 Safari/537.3",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:76.0) Gecko/20100101 Firefox/76.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edg/91.0.864.37",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) Presto/2.12.388 Version/12.18",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 10; SM-G973U Build/QP1A.190711.020) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.93 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/89.0.4389.82 Safari/537.36",
]


# include filer and holding in log messages for debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s [%(filer)s] [%(holding)s]",
    handlers=[logging.FileHandler("LOGS/filinglog.txt", mode="w")]
)
logger = logging.getLogger(__name__)

# ▶ The context container
from contextvars import ContextVar
_logctx = ContextVar("_logctx", default={"filer": "-", "holding": "-"})

def set_log_ctx(*, filer: str | None = None, holding: str | None = None):
    """
    Update the per-thread logging context.
    Omit either arg to keep its previous value.
    """
    cur = _logctx.get().copy()
    if filer   is not None: cur["filer"]   = filer
    if holding is not None: cur["holding"] = holding
    _logctx.set(cur)

# ▶ Custom record factory that injects the context into every LogRecord
_old_factory = logging.getLogRecordFactory()
def _factory(*a, **kw):
    rec = _old_factory(*a, **kw)
    ctx = _logctx.get()
    rec.filer   = ctx["filer"]
    rec.holding = ctx["holding"]
    return rec
logging.setLogRecordFactory(_factory)


# --- Initialize feature database ---
def init_feature_db():
    # remove if exists
    if os.path.exists(FEATURE_DB):
        os.remove(FEATURE_DB)
    conn = duckdb.connect(FEATURE_DB)
    conn.execute("""
    CREATE TABLE train_features (
        filer TEXT,
        filing_date TIMESTAMP,
        holding TEXT,
        metrics JSON,
        PRIMARY KEY (filer, filing_date, holding)
    )""")
    conn.execute("""
    CREATE TABLE test_features (
        filer TEXT,
        filing_date TIMESTAMP,
        holding TEXT,
        metrics JSON,
        PRIMARY KEY (filer, filing_date, holding)
                 
    )""")
    conn.close()

def load_grouped_filings():
    conn = duckdb.connect(SOURCE_DB)
    # join filings & holdings, group by filer
    df = conn.execute("""
        SELECT h.id AS filing_id,
               f.filer_name,
               f.date     AS filing_date,
               h.nameOfIssuer AS holding,
               h.sshPrnamt AS shares,
               h.value
        FROM holdings h
        JOIN filings f ON h.id = f.filing_id
        ORDER BY f.date
    """).df()
    conn.close()
    # group into list of (filer, sub_df)
    grouped = [(f, g.drop(columns=['filer_name']).reset_index(drop=True))
               for f, g in df.groupby('filer_name')]
    return grouped

con = duckdb.connect(SOURCE_DB)
con.execute("""
CREATE TABLE IF NOT EXISTS ticker_cache (
  company TEXT PRIMARY KEY,
  info JSON
);
""")
con.close()


def upsert_row(table, row):
    """
    row = (filer, filing_date, holding, metrics_json)
    """
    with duckdb.connect(FEATURE_DB) as con:
        con.execute(f"""
        INSERT INTO {table} (filer, filing_date, holding, metrics)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (filer, filing_date, holding)
        DO UPDATE SET metrics = EXCLUDED.metrics
        """, row)


def _json_to_df(txt: str | None) -> pd.DataFrame | None:
    """Rebuild DataFrame that was stored with df.to_json(orient='split')."""
    if not txt:                       # empty cell in cache
        return None
    # Wrap the literal string so read_json thinks it's reading a file
    return pd.read_json(StringIO(txt), orient="split")

def cacheGet(company: str):
    with duckdb.connect(SOURCE_DB) as con:
        row = con.execute(
            "SELECT info FROM ticker_cache WHERE company = ?", [company]
        ).fetchone()

    if not row:
        return None

    obj = json.loads(row[0])                       # ← stored string → dict
    info_dict = obj["info"]
    qis_df    = _json_to_df(obj["quarterly_income_stmt"])
    logger.info("💾 Cache get sucessful")
    return (info_dict, qis_df)

def _df_to_json(df: pd.DataFrame | None) -> str | None:
    """DataFrame → compact JSON (DuckDB stores it as TEXT)."""
    if df is None or df.empty:
        return None
    # 'split' keeps index/columns/data separately; easy to rebuild later
    return df.to_json(orient="split")


def cachePut(company: str, info_tuple):
    info_dict, qis_df = info_tuple
    payload = {
        "info": info_dict,
        "quarterly_income_stmt": _df_to_json(qis_df)
    }
    j = json.dumps(payload, separators=(",", ":"))   # compact string
    con = duckdb.connect(SOURCE_DB)
    # Upsert into the cache table
    con.execute("""
      INSERT INTO ticker_cache (company, info)
      VALUES (?, ?)
      ON CONFLICT (company) DO UPDATE SET info = EXCLUDED.info
    """, [company, j])
    con.close()


def backoffFn(fn, *args, **kwargs):
    """
    Run `fn`(*args, **kwargs) with exponential back-off on HTTP 401/429.
    """
    delays = itertools.chain([1, 2, 4, 8, 16], itertools.repeat(30))  # secs
    for delay in delays:
        try:
            return fn(*args, **kwargs)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (401, 429):
                logger.info(f"Rate-limited ({e.response.status_code}); sleeping {delay}s")
                time.sleep(delay)


def tickerLookup(company):
    try:

        q = urllib.parse.quote(company.strip().partition('/')[0])
        url = (
            f"https://query1.finance.yahoo.com/v1/finance/search?"
            f"q={q}&quotesCount=10&newsCount=0"
        )

        header = {"User-Agent": random.choice(HEADERS)}

        # Yahoo requires a User‑Agent or it 403s occasionally
        resp = requests.get(url, headers=header)
        if resp.status_code == 429:
            raise Exception(" in tickerLookup:\n         Rate limit exceeded: 429 Too Many Requests\n")
        resp.raise_for_status()

        data = resp.json()
        quotes = data.get("quotes", [])

        if not quotes:
            logger.error(f"❌ TICKER SYMBOL NOT FOUND on web")
            return None

        # The first hit is Yahoo’s best guess
        return quotes[0]["symbol"]

    except Exception as e:
        logger.error(f"{str(e)}")
        return None
    

def _sanitize(name: str) -> str:
    """Lower-case and remove periods for robust comparison."""
    return re.sub(r"\.", "", name).lower().strip()
    
def findTicker(company: str):
    mapping = TICKERMAPPING           # your JSON mapping already loaded
    target   = _sanitize(company)

    # 1) exact match (period-insensitive, case-insensitive)
    for rec in mapping.values():
        if _sanitize(rec["title"]) == target:
            return rec["ticker"]

    # 2) fuzzy match on sanitized titles
    sanitized_titles = { _sanitize(rec["title"]): rec["ticker"] 
                         for rec in mapping.values() }
    close = get_close_matches(target, sanitized_titles.keys(), n=1, cutoff=0.8)
    if close:
        return sanitized_titles[close[0]]

    logger.error(
        f"❌ TICKER SYMBOL NOT FOUND mapping file {TICKERS}"
    )
    return None



def getTickerSymbol(company):
    """ converts company name string to ticker """

    ticker = findTicker(company) # check if ticker is in the mapping file
    if ticker is None:
        ticker = backoffFn(tickerLookup, company) # if not found in mapping, try to look it up on the web
        if ticker is None:
            ticker = backoffFn(tickerLookup, company[:-3]) # try to remove last 3 chars (e.g. ' Inc' or ' Ltd') and look it up again
    if ticker:
        logger.info(f"TICKER SYMBOL FOUND")

    return ticker

    

def getOpenDays(date):
    """
    Get the NYSE market day that is BACKPERIOD days before `date`
    and the market day that is TESTPERIOD days after `date`.
    All in UTC.
    """
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(
        start_date=date - pd.Timedelta(days=int(BACKPERIOD * 3)),
        end_date=date + pd.Timedelta(days=int(TESTPERIOD * 3))
    )

    # Ensure the schedule's index is timezone-aware (UTC)
    past_days = []
    future_days = []
    for pdt in schedule.index:
        if pdt < date: past_days.append(pdt)
        if pdt > date: future_days.append(pdt)

    last_market_day = past_days[-1 - BACKPERIOD]
    next_market_day = future_days[TESTPERIOD]

    return last_market_day, next_market_day

def getNumShares(start, quartIncomeStmt):
    ''' Calculates the number of shares issued by a company at a specific time for 
        market cap calculation
    '''
    df = quartIncomeStmt
    if df is None or df.empty:
        logger.error("❌ No quarterly income statement data available.")
        return None

    def parse_date_from_col(col_name):
        # Some columns might look like '2023-07-31 yoy' or '2023-07-31 ttm'.
        # We'll split on space and parse the first token.
        raw_date = col_name.split(' ')[0]
        try:
            return pd.to_datetime(raw_date)  # This returns a tz-naive timestamp.
        except Exception:
            return None

    # Convert 'start' to a datetime, then remove timezone info (if any)
    start_date = pd.to_datetime(start)
    if start_date.tzinfo is not None:
        start_date = start_date.tz_localize(None)

    # We only care about columns whose parsed date is <= start_date.
    valid_cols = []
    for col in df.columns:
        cdate = parse_date_from_col(str(col))
        if cdate is not None and cdate <= start_date:
            valid_cols.append((col, cdate))
    if not valid_cols:
        logger.info("❌ No quarterly data columns are on or before " + str(start_date))
        return None

    # We'll attempt both 'Basic Average Shares' and 'Diluted Average Shares's
    possible_rows = ["Basic Average Shares", "Diluted Average Shares"]
    for col, cdate in valid_cols:
        for row_label in possible_rows:
            if row_label in df.index:
                val = df.loc[row_label, col]
                if not pd.isna(val):
                    return val

    logger.error('❌ No valid number of shares found.')
    return None


def getTestReturn(start, end, ticker):
    """Gets the percentage returns of the equity after the announcement."""
    # Convert to UTC if not already

    data = yf.download(ticker, start=start, end=end, interval="1h", auto_adjust=True, progress=False)
    if data.empty:
        return []

    realprice = [float(i[0]) for i in data['Open'].values.reshape(-1,1)]
    if not realprice:
        return []

    first_price = realprice[0]
    percentage_returns = [((price - first_price) / first_price) * 100 for price in realprice][1:]
    return percentage_returns
    
def getCurrentPrice(ticker, filingdate, lookback_days=7):
    """
    Get the last available open price before the filing date.
    Keep everything in UTC.
    """
    try:
        if filingdate.tzinfo is None:
            filingdate = filingdate.tz_localize('UTC')
        else:
            filingdate = filingdate.tz_convert('UTC')

        start_date = filingdate - pd.Timedelta(days=lookback_days)
        data = yf.download(ticker, start=start_date, end=filingdate, interval="1h", auto_adjust=True, progress=False)
        if data.empty or 'Open' not in data.columns:
            logger.error(f"No data found for {ticker} between {start_date} and {filingdate}")
            return None

        return data['Open'].values[-1][0]
    except ValueError as e:
        logger.error(f"yfinance download error for {ticker}: {e}")
        return None
    

def getTrainPriceInfo(start, filingdate, ticker, ticker_data=None):
    ''' Gets stock market cap and average volume over specified interval '''

    data = yf.download(ticker, start=start, end=filingdate, interval="1d", auto_adjust=True, progress=False)

    if data.empty:
        logger.error(f"❌ NO STOCK DATA FOUND for the given period {start} to {filingdate}")
        return None, None, None
    
    if ticker_data is not None:
        # get the number of shares at the time
        shares = getNumShares(start, ticker_data[1])
        if shares is None or pd.isna(shares):
            return None, None, None
    else:
        shares = 1 # for benchmark 

    # get average volume and last available open price before the filing date and calculate market cap
    avgVol = np.mean([item[0] for item in data['Volume'].values.tolist()][-BACKPERIOD:])
    lastprice = data['Close'].values.reshape(-1,1)[-1][0]

    if shares is None or pd.isna(shares):
        return None, None, None
    lastMktCap = lastprice * shares

    currentPrice = getCurrentPrice(ticker, filingdate)
    if currentPrice is None:
        logger.error(f"❌ No current price found for holding on {filingdate}")
        return None, None, None
    currentMktCap = currentPrice * shares # shares is 1 for benchmark

    return avgVol, lastMktCap, currentMktCap
    
def getTickerData(ticker):
    try:
        ticker_obj = backoffFn(
            yf.Ticker, ticker
        )
        return (ticker_obj.info, ticker_obj.quarterly_income_stmt)
    except Exception as exc:
        logger.info(f"Failed to fetch ticker data: {exc}")
        return None


def getStockInfo(holding, filingdate, filer, prcntChng, diffShares, diffDollars):
    ''' 
    Gets stock info for testing a traiing for a given holding
        TRAIN:
            Holding info: (Market cap, Volume, EBIDTA, Debt-Cash)
            Benchmark: (Market cap, Volume)
            Filer info: (Market cap, Volume, 
                        [of holding]: % change in shares, difference in shares, difference in dollars)

        TEST:
            Holding Market Cap and Volume
    '''

    trainInfo = {}
    testInfo = {}

    # Find the open market days before and after the filing date
    filingdate = filingdate.tz_localize(None)
    before, after = getOpenDays(filingdate)

    # HOLDING INFO
    ticker = getTickerSymbol(holding)
    if not ticker:
        logger.error(f'❌ TICKER SYMBOL NOT FOUND for holding')
        return None, None

    # to get total debt and cash we use yfinance ticker object
    # backoff is implemented to avoid rate limiting
    ticker_data = cacheGet(ticker)
    if ticker_data is None:
        ticker_data = getTickerData(ticker)
        if ticker_data is None:
            logger.error(f'❌ TICKER DATA NOT FOUND for holding')
            return None, None
        cachePut(ticker, ticker_data)
    
    logger.info(f"TICKER DATA FOUND for holding")

    debt = ticker_data[0].get('totalDebt', None)
    cash = ticker_data[0].get('totalCash', None)

    holdingAvgVol, holdingLastMktCap, holdingCurrentMktCap = getTrainPriceInfo(before, filingdate, ticker, ticker_data=ticker_data)
    if holdingAvgVol is None or holdingLastMktCap is None or holdingCurrentMktCap is None:
        logger.error(f"❌ NO STOCK DATA FOUND for holding for the given period {before} to {filingdate}")
        return None, None
    
    trainInfo['holdingAvgVol'] = holdingAvgVol
    trainInfo['holdingLastMktCap'] = holdingLastMktCap
    trainInfo['currentMktCap'] = holdingCurrentMktCap

    trainInfo['EBITDA'] = ticker_data[0].get('ebitda', None)
    trainInfo['Debt-Cash'] = debt - cash if debt is not None and cash is not None else None

    # BENCHMARK INFO (S&P 500)
    benchmarkAvgVol, benchmarkLastPrice, benchmarkCurrentPrice = getTrainPriceInfo(before, filingdate, '^GSPC')
    if benchmarkAvgVol is None or benchmarkLastPrice is None or benchmarkCurrentPrice is None:
        logger.error(f"❌ NO STOCK DATA FOUND for S&P 500 for the given period {before} to {filingdate}")
        return None, None
    trainInfo['benchmarkAvgVol'] = benchmarkAvgVol
    trainInfo['benchmarkLastPrice'] = benchmarkLastPrice
    trainInfo['benchmarkCurrentPrice'] = benchmarkCurrentPrice

    # FILER INFO
    filerTicker = getTickerSymbol(filer)
    if not filerTicker:
        logger.error(f'❌ TICKER SYMBOL NOT FOUND for FILER')
        return None, None

    filerAvgVol, filerLastMktCap, filerCurrentMktCap = getTrainPriceInfo(before, filingdate, ticker, ticker_data=ticker_data)
    if filerAvgVol is None or filerLastMktCap is None or filerCurrentMktCap is None:
        logger.error(f"❌ NO STOCK DATA FOUND for filer for the given period {before} to {filingdate}")
        return None, None
    
    trainInfo['filerAvgVol'] = filerAvgVol
    trainInfo['filerLastMktCap'] = filerLastMktCap
    trainInfo['filerCurrentMktCap'] = filerCurrentMktCap
    trainInfo['filerEBITDA'] = ticker_data[0].get('ebitda', None)
    trainInfo['filerDebt-Cash'] = debt - cash if debt is not None and cash is not None else None

    trainInfo['percentChng'] = prcntChng
    trainInfo['diffShares'] = diffShares # how many more or less shares (negative if less than prev filing)
    trainInfo['diffDollars'] = diffDollars # "
 
    testInfo['RealPrice'] = getTestReturn(filingdate, after, ticker)
    if testInfo is None:
        logger.error(f"❌ NO STOCK DATA for holding for the given period {filingdate} to {after}")
        return None, None
    
    return trainInfo, testInfo


def _process_single_filer(filer, df_group):
    '''
        Iterates through filings for a single filer, extracting train/test features.
         - Skips the first filing (no previous data to compare)
         - For each subsequent filing, compares holdings to the previous one
         - Detemines if a holding was added, removed, or changed
         - For each change, retrieves stock info and prepares train/test data  
    '''
    set_log_ctx(filer=filer, holding="-") 
    logger.info("+++++++++BEGIN PROCESSING+++++++++")

    prev = {}  # mapping holding->(shares,value)
    for _, row in df_group.iterrows():
        date = row['filing_date']
        holding = row['holding']
        shares = row['shares']
        value = row['value']

        # first filing: init prev
        if not prev:
            prev = {holding:(shares,value)}
            continue

        # compare to prev holdings
        for h,(old_shares,old_val) in list(prev.items()):
            set_log_ctx(holding=h) 

            if h not in df_group['holding'].values:
                # full liquidation
                prcnt, dsh, dv = -1, -old_shares, -old_val
                train, test = getStockInfo(h, date, filer, prcnt, dsh, dv)
                if train and test:
                    upsert_row('train_features',
                               (filer, date, h, json.dumps(train)))
                    upsert_row('test_features',
                               (filer, date, h, json.dumps(test)))

        cur = {row['holding']:(row['shares'],row['value']) 
               for _,row in df_group[df_group['filing_date']==date].iterrows()}
        
        for h,(cur_sh,cur_val) in cur.items():
            set_log_ctx(holding=h)

            if 'ETF' in h or 'TRUST' in h or 'FUND' in h:
                logger.info(f"Skipping non stock: {h}")
                continue

            old_sh,old_val = prev.get(h,(0,0)) # if not in prev, assume 0 shares and 0 value

            if cur_sh == old_sh: # No change. I am worried that stock movements resulting from a company holding a contenious position could skew the model.
                continue       
            if old_sh==0: # bought new holding
                prcnt, dsh, dv = 1, cur_sh, cur_val-old_val
            elif cur_sh>old_sh: # bought some more
                prcnt, dsh, dv = cur_sh/old_sh, cur_sh-old_sh, cur_val-old_val
            else: # sold some but not all
                prcnt, dsh, dv = (cur_sh/old_sh), cur_sh-old_sh, cur_val-old_val

            # get stock info for train/test
            train, test = getStockInfo(h, date, filer, prcnt, dsh, dv)

            if train and test:
                upsert_row('train_features',
                           (filer, date, h, json.dumps(train)))
                upsert_row('test_features',
                           (filer, date, h, json.dumps(test)))
                logger.info("✅ Holding processed")
                
        prev = cur

    set_log_ctx(filer=filer, holding="-") 
    logger.info("+++++++++END PROCESSING+++++++++")


# --- Main orchestration ---
def main():
    start = time.time()
    init_feature_db()
    grouped = load_grouped_filings()
    print(f"Found {len(grouped)} filers to process")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:

        futures = {pool.submit(_process_single_filer, f, df): f for f,df in grouped}

        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Processing filers"):
            try:
                fut.result()          # ← catches thread exceptions
            except Exception as exc:
                logger.exception("Filer failed: %s", exc)
                continue

    duration = time.time()-start
    print(f"Completed in {duration:.1f}s")

if __name__ == '__main__':
    main()
