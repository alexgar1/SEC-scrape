import pandas as pd
import sys
import datetime, time
import yfinance as yf
import requests
import pandas_market_calendars as mcal
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import json
from bs4 import BeautifulSoup
from tqdm import tqdm
import random
import shelve
import urllib.parse



# TODO
# Ticker lookup is failing for unkown reason
# Include open price day of filing in training data

### modify getCurrentPrice to look at next open price if filing date is outside of market hours


TEST = False
if not TEST:
    FILINGDB = 'DATA/filings.feather'
    HOLDINGDB = 'DATA/holdings.feather'
    TRAINJSON = 'DATA/X'
    TESTJSON = 'DATA/y'
else:
    FILINGDB = 'DATA/TESTfilings.feather'
    HOLDINGDB = 'DATA/TESTholdings.feather'
    TRAINJSON = 'DATA/TESTX'
    TESTJSON = 'DATA/TESTy'

# A random header is chosen to avoid 403 errors
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

BACKPERIOD = 10 # DAYS to look back and forward for stock data
TESTPERIOD = 3 # DAYS post annoucement to assess a price movement (7 hours in normal market day).
LOG = 'filinglog.txt'
CACHE = shelve.open("DATA/tickerCache")

session = requests.Session()

# clean up
open(LOG, 'w').close()

def writelog(msg):
    with open(LOG, 'a') as f:
        f.write(msg + '\n')

def saveJson(data, tablename):
    def default_converter(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
    
    with open(tablename+".json", "w") as f:
        json.dump(data, f, default=default_converter)


def getTicker(company, verbose=False):
    """ converts company name string to ticker """
    # try:
    #     yfinance_url = "https://query2.finance.yahoo.com/v1/finance/search"
    #     user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    #     params = {"q": company, "quotes_count": 1, "country": "United States"}
        
    #     res = requests.get(url=yfinance_url, params=params, headers=HEADERS)
    #     data = res.json()
        
    #     if "quotes" in data and len(data["quotes"]) > 0:
    #         if verbose: print(f"Found via Yahoo Finance API: {data['quotes'][0]['symbol']}")
    #         return data['quotes'][0]['symbol']
    # except Exception as e:
    #     if verbose: print(f"Yahoo Finance API failed: {str(e)}")

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
            writelog(f"No matches returned for: {company}")
            return None

        # The first hit is Yahoo’s best guess
        return quotes[0]["symbol"]

    except Exception as e:
        writelog(f"\n   !!!ERROR:\n         {str(e)}")
        return None
    

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
        writelog("\n   !!!ERROR:\n         No quarterly income statement data available.")
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
        writelog("No quarterly data columns are on or before " + str(start_date))
        return None

    # We'll attempt both 'Basic Average Shares' and 'Diluted Average Shares's
    possible_rows = ["Basic Average Shares", "Diluted Average Shares"]
    for col, cdate in valid_cols:
        for row_label in possible_rows:
            if row_label in df.index:
                val = df.loc[row_label, col]
                if not pd.isna(val):
                    return val

    writelog('\n   !!!ERROR:\n         No valid number of shares found.')
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
        writelog(f"\n   !!!ERROR:\n         No stock data found for {ticker} for the given period {start} to {filingdate}")
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

    print(shares, lastprice)
    if shares is None or pd.isna(shares):
        print(shares, lastprice)
        return None, None, None
    lastMktCap = lastprice * shares

    currentPrice = getCurrentPrice(ticker, filingdate)
    if currentPrice is None:
        writelog(f"\n   !!!ERROR:\n         No current price found for {ticker} on {filingdate}")
        return None, None, None
    currentMktCap = currentPrice * shares # shares is 1 for benchmark

    return avgVol, lastMktCap, currentMktCap
    
def getTickerData(ticker):
    max_attempts = 5
    attempt = 0
    delay = 1
    while attempt < max_attempts:
        try:
            session.headers['User-agent'] = random.choice(HEADERS)
            ticker_obj = yf.Ticker(ticker, session=session)
            if ticker_obj is None:
                writelog(f"\n   !!!ERROR:\n         No ticker object found for {ticker}")
                return None, None
            
            # Use the ticker object to get the info and quarterly income statement
            ticker_data = (ticker_obj.info, ticker_obj.quarterly_income_stmt)
            CACHE[ticker] = ticker_data
            CACHE.sync()
            return ticker_data

        except Exception as e:
            attempt += 1
            writelog(f"Attempt {attempt} for ticker {ticker} failed due to: {str(e)}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # exponential backoff
    else:
        writelog(f"\n   !!!ERROR:\n         Unable to retrieve ticker data for {ticker} after {max_attempts} attempts due to rate limiting.")
        return None, None


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
        writelog(f'\n   !!!ERROR:\n         ticker for HOLDING: "{holding}" not found')
        return None, None

    # to get total debt and cash we use yfinance ticker object
    # backoff is implemented to avoid rate limiting
    if ticker in CACHE:
        ticker_data = CACHE[ticker]
    else:
        ticker_data = getTickerData(ticker)
        if None in ticker_data:
            writelog(f"\n   !!!ERROR:\n         No ticker info found for {ticker}")
            return None, None

    debt = ticker_data[0].get('totalDebt', None)
    cash = ticker_data[0].get('totalCash', None)

    holdingAvgVol, holdingLastMktCap, holdingCurrentMktCap = getTrainPriceInfo(before, filingdate, ticker, ticker_data=ticker_data)
    if holdingAvgVol is None or holdingLastMktCap is None or holdingCurrentMktCap is None:
        writelog(f"\n   !!!ERROR:\n         No stock data found for {ticker} for the given period {before} to {filingdate}")
        return None, None
    
    trainInfo['holdingAvgVol'] = holdingAvgVol
    trainInfo['holdingLastMktCap'] = holdingLastMktCap
    trainInfo['currentMktCap'] = holdingCurrentMktCap

    trainInfo['EBITDA'] = ticker_data[0].get('ebitda', None)
    trainInfo['Debt-Cash'] = debt - cash if debt is not None and cash is not None else None

    # BENCHMARK INFO (S&P 500)
    benchmarkAvgVol, benchmarkLastPrice, benchmarkCurrentPrice = getTrainPriceInfo(before, filingdate, '^GSPC')
    if benchmarkAvgVol is None or benchmarkLastPrice is None or benchmarkCurrentPrice is None:
        writelog(f"\n   !!!ERROR:\n         No stock data found for S&P 500 for the given period {before} to {filingdate}")
        return None, None
    trainInfo['benchmarkAvgVol'] = benchmarkAvgVol
    trainInfo['benchmarkLastPrice'] = benchmarkLastPrice
    trainInfo['benchmarkCurrentPrice'] = benchmarkCurrentPrice


    # FILER INFO
    filerTicker = getTicker(filer)
    if not filerTicker:
        writelog(f'\n   !!!ERROR:\n         ticker for FILER: "{filer}" not found')
        return None, None

    filerAvgVol, filerLastMktCap, filerCurrentMktCap = getTrainPriceInfo(before, filingdate, ticker, ticker_data=ticker_data)
    if filerAvgVol is None or filerLastMktCap is None or filerCurrentMktCap is None:
        writelog(f"\n   !!!ERROR:\n         No stock data found for {ticker} for the given period {before} to {filingdate}")
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
        writelog(f"\n   !!!ERROR:\n         No stock data found for {ticker} for the given period {filingdate} to {after}")
        return None, None
    
    return trainInfo, testInfo


def parseData():
    try:
        df_filings = pd.read_feather(FILINGDB)
        df_holdings = pd.read_feather(HOLDINGDB)
    except Exception as e:
        writelog(f"ERROR reading .feather files:{e}")
        sys.exit(1)

    if 'shares' not in df_holdings.columns:
        df_holdings['shares'] = df_holdings['shrsOrPrnAmt'].apply(
            lambda x: x['sshPrnamt'] if isinstance(x, dict) and 'sshPrnamt' in x else 0
        )

    merged = df_holdings.merge(
        df_filings,
        left_on='id',
        right_on='filing_id',
        how='inner',
        suffixes=('_holding','_filing')
    )

    merged.sort_values(by=['filer_name', 'date'], inplace=True)
    grouped = merged.groupby('filer_name', group_keys=True)

    return grouped

def getModelData(holding, filing_date, filer, prcnt, diffShares, diffDollar, X, y):
    '''
    Data format:
    train/test 
        -> filer 
            -> filing date
                -> holding
    '''
    train, test = getStockInfo(holding, filing_date, filer, prcnt, diffShares, diffDollar)
    if not train or not test:
        return X, y

    filing_date_str = filing_date.strftime('%Y-%m-%d %H:%M:%S %Z')
    X.setdefault(filer, {}).setdefault(filing_date_str, {})[holding] = train
    y.setdefault(filer, {}).setdefault(filing_date_str, {})[holding] = test

    return X, y


def inferAction(grouped):
    X = {}
    y = {}

    for filer, group_df in tqdm(grouped):
        group_df = group_df.sort_values(by='date')

        prev_holdings = {}
        prev_values   = {}
        prev_date = None

        writelog('='*5 + filer + '='*5)

        for (filing_date, filing_id), filing_rows in tqdm(group_df.groupby(['date','filing_id'])):

            # print(f"\n" + '='*5 + f" {filer} {filing_date} " + '='*5)

            filing_date = filing_date.tz_localize('UTC')
            
            writelog(f"\nFiling date: {filing_date}  (Filing ID: {filing_id})")
            current_holdings = {}
            current_values   = {}

            for _, row in filing_rows.iterrows():
                holding = row['nameOfIssuer']
                shares  = row['shares']
                value   = row['value']

                current_holdings[holding] = shares
                current_values[holding]   = value    # ← store current value

            if prev_date is not None:
                # 1) handle buys, sells, increases/decreases
                for holding, curShares in current_holdings.items():
                    oldShares  = prev_holdings.get(holding, 0)
                    oldValue   = prev_values.get(holding, 0)
                    value_diff = current_values[holding] - oldValue

                    if holding not in prev_holdings:
                        writelog(f"  {filer} BOUGHT NEW {curShares} shares of {holding}")
                        X, y = getModelData(
                            holding, filing_date, filer,
                            1,                   # ratio
                            curShares,           # share diff
                            value_diff,          # ← dollar diff
                            X, y
                        )

                    elif curShares > oldShares:
                        diff = curShares - oldShares
                        writelog(f"  {filer} BOUGHT MORE {diff} shares of {holding}")
                        X, y = getModelData(
                            holding, filing_date, filer,
                            curShares/oldShares, # ratio
                            diff,                # share diff
                            value_diff,          # ← dollar diff
                            X, y
                        )

                    elif curShares < oldShares:
                        diff = curShares - oldShares
                        writelog(f"  {filer} SOLD SOME {-diff} shares of {holding}")
                        if oldShares == 0:
                            writelog(f"  {filer} BOUGHT NEW {curShares} shares of {holding}")
                            X, y = getModelData(
                                holding, filing_date, filer,
                                1,        
                                curShares, 
                                value_diff,
                                X, y
                            )
                        X, y = getModelData(
                            holding, filing_date, filer,
                            curShares/oldShares, # ratio
                            diff,                # share diff (negative)
                            value_diff,          # ← dollar diff (negative)
                            X, y
                        )

                # 2) handle full sales
                for holding, oldShares in prev_holdings.items():
                    if holding not in current_holdings:
                        oldValue   = prev_values.get(holding, 0)
                        value_diff = - oldValue
                        writelog(f"  {filer} SOLD ALL {oldShares} shares of {holding}")
                        X, y = getModelData(
                            holding, filing_date, filer,
                            -1,                  # ratio
                            -oldShares,          # share diff
                            value_diff,          # ← dollar diff
                            X, y
                        )

            prev_holdings = current_holdings
            prev_values   = current_values
            prev_date = filing_date

    saveJson(X, TRAINJSON)
    saveJson(y, TESTJSON)


def main():
    start_time = time.time()

    data = parseData()
    inferAction(data)
    CACHE.close()

    end_time = time.time()
    writelog("Total execution time:" + str(end_time - start_time))


if __name__ == "__main__":
    main()
