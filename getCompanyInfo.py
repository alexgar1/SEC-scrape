
import pandas as pd
import time
import yfinance as yf
import requests
import pandas_market_calendars as mcal
import numpy as np
import random
import urllib.parse
import yfinance.utils as yfu
import itertools
import duckdb
import json
import logging
import sys
import threading

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

_YF_SESSION = None
_YF_SESSION_LOCK = threading.Lock()

def shareYFsession() -> requests.Session:
    global _YF_SESSION
    with _YF_SESSION_LOCK:
        if _YF_SESSION is None:
            _YF_SESSION = requests.Session()
            _YF_SESSION.headers.update({"User-Agent": random.choice(HEADERS)})
            yfu.get_yf_crumb_and_cookies(session=_YF_SESSION)
        return _YF_SESSION

BACKPERIOD = 10 # DAYS to look back and forward for stock data
TESTPERIOD = 3 # DAYS post annoucement to assess a price movement (7 hours in normal market day).

CACHE_DB = 'DATA/sec_filings.duckdb'

con = duckdb.connect(CACHE_DB)
con.execute("""
CREATE TABLE IF NOT EXISTS ticker_cache (
  company TEXT PRIMARY KEY,
  info JSON
);
""")
con.close()

# Configure root logger once
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s %(message)s",
    handlers=[
        logging.FileHandler("LOGS/filinglog.txt", mode="w"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)



def cacheGet(company: str):
    con = duckdb.connect(CACHE_DB)
    # Returns Python string of JSON, or None
    result = con.execute(
        "SELECT info FROM ticker_cache WHERE company = ?", [company]
    ).fetchone()
    con.close()
    return json.loads(result[0]) if result else None

def cachePut(company: str, info_obj):
    j = json.dumps(info_obj)
    con = duckdb.connect(CACHE_DB)
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
                # refresh crumb/cookies once in case Yahoo invalidated them
                with _YF_SESSION_LOCK:
                    global _YF_SESSION
                    _YF_SESSION = None
                continue
            raise


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
            raise Exception("\n   !!!ERROR in getTicker:\n         Rate limit exceeded: 429 Too Many Requests\n")
        resp.raise_for_status()

        data = resp.json()
        quotes = data.get("quotes", [])

        if not quotes:
            return None

        # The first hit is Yahoo’s best guess
        return quotes[0]["symbol"]

    except Exception as e:
        logger.error(f"{str(e)}")
        return None



def getTicker(company):
    """ converts company name string to ticker """

    ticker = tickerLookup(company)
    if ticker is None:
        ticker = tickerLookup(company[:-3])
        if ticker is None:
            logger.info(f"No matches returned for: {company}")
    
    return ticker

    

def getOpenDays(date):
    """
    Get the NYSE market day that is BACKPERIOD days before `date`
    and the market day that is TESTPERIOD days after `date`.
    All in UTC.
    """
    # Ensure `date` is in UTC
    if date.tzinfo is None:
        date = date.tz_localize('UTC')
    else:
        date = date.tz_convert('UTC')

    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(
        start_date=date - pd.Timedelta(days=int(BACKPERIOD * 3)),
        end_date=date + pd.Timedelta(days=int(TESTPERIOD * 3))
    )

    # Ensure the schedule's index is timezone-aware (UTC)
    valid_sessions = schedule.index.tz_localize('UTC')

    # Get the last market day that is strictly < date, then go BACKPERIOD back
    past_days = valid_sessions[valid_sessions < date]
    last_market_day = past_days[-1 - BACKPERIOD]

    # Get the next market day that is strictly > date, then go TESTPERIOD forward
    future_days = valid_sessions[valid_sessions > date]
    next_market_day = future_days[TESTPERIOD]

    return last_market_day, next_market_day

def getNumShares(start, quartIncomeStmt):
    ''' Calculates the number of shares issued by a company at a specific time for 
        market cap calculation
    '''
    df = quartIncomeStmt
    if df.empty:
        logger.error("No quarterly income statement data available.")
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
        logger.info("No quarterly data columns are on or before " + str(start_date))
        return None

    # We'll attempt both 'Basic Average Shares' and 'Diluted Average Shares's
    possible_rows = ["Basic Average Shares", "Diluted Average Shares"]
    for col, cdate in valid_cols:
        for row_label in possible_rows:
            if row_label in df.index:
                val = df.loc[row_label, col]
                if not pd.isna(val):
                    return val

    logger.error('No valid number of shares found.')
    return None


def getTestReturn(start, end, ticker):
    """Gets the percentage returns of the equity after the announcement."""
    # Convert to UTC if not already
    if start.tzinfo is None:
        start = start.tz_localize('UTC')
    else:
        start = start.tz_convert('UTC')

    if end.tzinfo is None:
        end = end.tz_localize('UTC')
    else:
        end = end.tz_convert('UTC')

    data = yf.download(ticker, start=start, end=end, interval="1h", auto_adjust=True, progress=False)
    print(data)
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
    if filingdate.tzinfo is None:
        filingdate = filingdate.tz_localize('UTC')
    else:
        filingdate = filingdate.tz_convert('UTC')

    start_date = filingdate - pd.Timedelta(days=lookback_days)
    data = yf.download(ticker, start=start_date, end=filingdate, interval="1h", auto_adjust=True, progress=False)
    if data.empty or 'Open' not in data.columns:
        return None

    return data['Open'].values[-1][0]
    



def getTrainPriceInfo(start, filingdate, ticker, ticker_data=None):
    ''' Gets stock market cap and average volume over specified interval '''

    if start.tzinfo is None:
        start = start.tz_localize('UTC')
    else:
        start = start.tz_convert('UTC')

    if filingdate.tzinfo is None:
        filingdate = filingdate.tz_localize('UTC')
    else:
        filingdate = filingdate.tz_convert('UTC')

    data = yf.download(ticker, start=start, end=filingdate, interval="1d", auto_adjust=True, progress=False)

    if data.empty:
        logger.error(f"No stock data found for {ticker} for the given period {start} to {filingdate}")
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
        logger.error(f"No current price found for {ticker} on {filingdate}")
        return None, None, None
    currentMktCap = currentPrice * shares # shares is 1 for benchmark

    return avgVol, lastMktCap, currentMktCap
    
def getTickerData(ticker):
    try:
        ticker_obj = backoffFn(
            yf.Ticker, ticker, session=shareYFsession()
        )
        return (ticker_obj.info, ticker_obj.quarterly_income_stmt)
    except Exception as exc:
        logger.info(f"Failed to fetch ticker data for {ticker}: {exc}")
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
    if filingdate.tzinfo is None:
        filingdate = filingdate.tz_localize('UTC')
    else:
        filingdate = filingdate.tz_convert('UTC')


    trainInfo = {}
    testInfo = {}

    # Find the open market days before and after the filing date
    filingdate = filingdate.tz_localize(None)
    before, after = getOpenDays(filingdate)

    # HOLDING INFO
    ticker = getTicker(holding)
    if not ticker:
        logger.info(f'\n   !!!ERROR:\n         ticker for HOLDING: "{holding}" not found')
        return None, None

    # to get total debt and cash we use yfinance ticker object
    # backoff is implemented to avoid rate limiting
    ticker_data = cacheGet(ticker)
    if ticker_data is None:
        ticker_data = getTickerData(ticker)
        if ticker_data is None:
            logger.error(f"No ticker info found for {ticker}")
            return None, None
        cachePut(ticker, ticker_data)

    debt = ticker_data[0].get('totalDebt', None)
    cash = ticker_data[0].get('totalCash', None)

    holdingAvgVol, holdingLastMktCap, holdingCurrentMktCap = getTrainPriceInfo(before, filingdate, ticker, ticker_data=ticker_data)
    if holdingAvgVol is None or holdingLastMktCap is None or holdingCurrentMktCap is None:
        logger.error(f"No stock data found for {ticker} for the given period {before} to {filingdate}")
        return None, None
    
    trainInfo['holdingAvgVol'] = holdingAvgVol
    trainInfo['holdingLastMktCap'] = holdingLastMktCap
    trainInfo['currentMktCap'] = holdingCurrentMktCap

    trainInfo['EBITDA'] = ticker_data[0].get('ebitda', None)
    trainInfo['Debt-Cash'] = debt - cash if debt is not None and cash is not None else None

    # BENCHMARK INFO (S&P 500)
    benchmarkAvgVol, benchmarkLastPrice, benchmarkCurrentPrice = getTrainPriceInfo(before, filingdate, '^GSPC')
    if benchmarkAvgVol is None or benchmarkLastPrice is None or benchmarkCurrentPrice is None:
        logger.error(f"No stock data found for S&P 500 for the given period {before} to {filingdate}")
        return None, None
    trainInfo['benchmarkAvgVol'] = benchmarkAvgVol
    trainInfo['benchmarkLastPrice'] = benchmarkLastPrice
    trainInfo['benchmarkCurrentPrice'] = benchmarkCurrentPrice


    # FILER INFO
    filerTicker = getTicker(filer)
    if not filerTicker:
        logger.info(f'\n   !!!ERROR:\n         ticker for FILER: "{filer}" not found')
        return None, None

    filerAvgVol, filerLastMktCap, filerCurrentMktCap = getTrainPriceInfo(before, filingdate, ticker, ticker_data=ticker_data)
    if filerAvgVol is None or filerLastMktCap is None or filerCurrentMktCap is None:
        logger.error(f"No stock data found for {ticker} for the given period {before} to {filingdate}")
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
        logger.error(f"No stock data found for {ticker} for the given period {filingdate} to {after}")
        return None, None
    
    return trainInfo, testInfo