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




# TODO
# Include open price day of filing in training data


HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "max-age=0",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
}
BACKPERIOD = 10 # DAYS to look back and forward for stock data
TESTPERIOD = 3 # DAYS post annoucement to assess a price movement (7 hours in normal market day).
LOG = 'filinglog.txt'

# clean up
open(LOG, 'w').close()

def writelog(msg):
    with open(LOG, 'a') as f:
        f.write(msg + '\n')

def saveJson(data, tablename):
    with open(tablename+".json", "w") as f:
        json.dump(data, f)



def getTicker(company, verbose=False):
    """
    Enhanced ticker lookup with multiple fallback strategies
    Returns: ticker (str) or None if not found
    """
    # Strategy 1: Original Yahoo Finance search
    try:
        yfinance_url = "https://query2.finance.yahoo.com/v1/finance/search"
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
        params = {"q": company, "quotes_count": 1, "country": "United States"}
        
        res = requests.get(url=yfinance_url, params=params, headers={'User-Agent': user_agent})
        data = res.json()
        
        if "quotes" in data and len(data["quotes"]) > 0:
            if verbose: print(f"Found via Yahoo Finance API: {data['quotes'][0]['symbol']}")
            return data['quotes'][0]['symbol']
    except Exception as e:
        if verbose: print(f"Yahoo Finance API failed: {str(e)}")





def getOpenDays(date):
    # Get price info for the marker day before and after the filing date
    date = date.tz_localize(None)
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=date - pd.Timedelta(days=int(BACKPERIOD*3)), end_date=date + pd.Timedelta(days=int(TESTPERIOD*3))) # start with broad calendar range
    last_market_day = schedule[schedule.index < date].index[-1-BACKPERIOD]
    next_market_day = schedule[schedule.index > date].index[TESTPERIOD]
    return last_market_day, next_market_day

def getNumShares(start, ticker_info):
    ''' Calculates the number of shares issued by a company at a specific time for 
        to be used for market cap calculation
    '''
    df = ticker_info.quarterly_income_stmt
    if df.empty:
        writelog("ERROR: No quarterly income statement data available.")
        return None

    def parse_date_from_col(col_name):
        # Some columns might look like '2023-07-31 yoy' or '2023-07-31 ttm'.
        # We'll split on space and parse the first token.
        raw_date = col_name.split(' ')[0]
        try:
            return pd.to_datetime(raw_date)
        except Exception:
            return None

    # We only care about columns whose parsed date is <= start (the filing/training date).
    start_date = pd.to_datetime(start)
    valid_cols = []
    for col in df.columns:
        cdate = parse_date_from_col(str(col))
        if cdate is not None and cdate <= start_date:
            valid_cols.append((col, cdate))
    if not valid_cols:
        writelog("No quarterly data columns are on or before", start_date)
        return None

    # We'll attempt both 'Basic Average Shares' and 'Diluted Average Shares'
    possible_rows = ["Basic Average Shares", "Diluted Average Shares"]
    for col, cdate in valid_cols:
        for row_label in possible_rows:
            # Check if row_label is actually in the DataFrame index
            if row_label in df.index:
                val = df.loc[row_label, col]
                # If not NaN, return it
                if not pd.isna(val):
                    return val

    writelog('ERROR: No valid number of shares found.')
    return None


def getPriceInfo(start, end, ticker, train, ticker_info=None):
    ''' Gets stock price data over specified interval '''
    if train: # if training get 
        data = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True)
    else:
        data = yf.download(ticker, start=start, end=end, interval="1h", auto_adjust=True)
    
    if data.empty:
        writelog(f"ERROR: No stock data found for {ticker} for the given period {start} to {end}")
        return None
    
    if ticker_info is not None:
        # get the number of shares at the time
        shares = getNumShares(start, ticker_info)
        if shares is None or pd.isna(shares):
            return None
    else:
        shares = 1 # for benchmark
    
    # get the market cap at each time
    # this allows for the accurate size of the company and its movements to be taken into account
    if train: # TRAIN data
        volume = [item[0] for item in data['Volume'].values.tolist()]
        realprice = [float(i[0] * shares) for i in data['Close'].values.reshape(-1,1)]
        realpriceLs = realprice[-BACKPERIOD:]
        volumeLs = volume[-BACKPERIOD:]
        priceInfo = realpriceLs, volumeLs, data['Close'].iloc[-1]
        return priceInfo
    
    else: # TEST data
        realprice = [float(i[0] * shares) for i in data['Open'].values.reshape(-1,1)]
        priceInfo = {'RealPrice': realprice[:TESTPERIOD]}
        return priceInfo



def getStockInfo(issuer, filingdate, filer, prcntChng, diffShares):
    ''' 
    Gets stock info for testing a traiing for a given issuer
        TRAIN:
            Holding info: (Market cap, Volume, EBIDTA, Debt-Cash)
            Benchmark: (Market cap, Volume)
            Filer info: (Market cap, Volume, 
                        [of holding]: % change in shares, difference in shares, difference in dollars)

        TEST:
            Holding Market Cap and Volume
    '''
    trainInfo = {} 

    ticker = getTicker(issuer)
    if not ticker:
        writelog(f'ERROR: ticker for "{issuer}" not found')
        return None, None
    
    ticker_info = yf.Ticker(ticker)

    # Find the open market days before and after the filing date
    before, after = getOpenDays(filingdate)

    # HOLDING INFO
    holdingInfo = getPriceInfo(before, filingdate, ticker, train=True, ticker_info=ticker_info)
    if holdingInfo is None:
        return None, None
    elif holdingInfo[0] is None or holdingInfo[1] is None:
        return None, None
    
    holdingPriceTrainData = holdingInfo[0]
    holdingVolume = holdingInfo[1]
    lastprice = holdingInfo[2]

    debt = ticker_info.info.get('totalDebt', None)
    cash = ticker_info.info.get('totalCash', None)

    # Convert keys in holdingPriceTrainData to strings
    trainInfo['holdingPriceTrainData'] = holdingPriceTrainData
    trainInfo['holdingVolume'] = holdingVolume
    trainInfo['EBITDA'] = ticker_info.info.get('ebitda', None)
    trainInfo['Debt-Cash'] = debt - cash if debt is not None and cash is not None else None

    # BENCHMARK INFO (S&P 500)
    benchmark = getPriceInfo(before, filingdate, '^GSPC', train=True)
    trainInfo['benchmarkPrice'] = benchmark[0]
    trainInfo['benchmarkVolume'] = benchmark[1]

    # FILER INFO
    filerTicker = getTicker(filer)
    writelog(filerTicker)
    filerInfo = getPriceInfo(before, filingdate, filerTicker, train=True, ticker_info=yf.Ticker(filerTicker))
    if filerInfo is None:
        return None, None
    if filerInfo[0] is None or filerInfo[1] is None:
        writelog(f"ERROR: No stock data found for filer {filerTicker} for the given period {before} to {filingdate}")
        return None, None
    trainInfo['filerPrices'] = filerInfo[0]
    trainInfo['filerVolume'] = filerInfo[1]
    trainInfo['percentChng'] = prcntChng
    trainInfo['diffShares'] = diffShares # how many more or less shares (negative if less)
    trainInfo['diffDollars'] = diffShares * float(lastprice.iloc[0]) # " in dollars

    testInfo = getPriceInfo(filingdate, after, ticker, train=False, ticker_info=ticker_info)
    if testInfo is None:
        writelog(f"ERROR: No stock data found for {ticker} for the given period {filingdate} to {after}")
        return None, None

    
    return trainInfo, testInfo


def parseData():
    try:
        df_filings = pd.read_feather('DATA/filings.feather')
        df_holdings = pd.read_feather('DATA/holdings.feather')
    except Exception as e:
        writelog("ERROR reading .feather files:", e)
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

def getModelData(issuer, filing_date, filer, prcnt, diff, X, y):
    '''
    Data format:

    train/test <- filer <- filing date <- issuer
    
    '''
    train, test = getStockInfo(issuer, filing_date, filer, prcnt, diff)
    filing_date = filing_date.strftime('%Y-%m-%d %X')
    
    if not train or not test:  # Ensure valid data before proceeding
        return X, y

    # Use setdefault() to simplify dictionary initialization
    try:
        X.setdefault(filer, {}).setdefault(filing_date, {})[issuer] = train
        y.setdefault(filer, {}).setdefault(filing_date, {})[issuer] = test
    except:
        print('ERROR building json for', filer, filing_date, issuer)
        return X, y

    return X, y


def inferAction(grouped):
    X = {}
    y = {}

    for filer, group_df in grouped:
        group_df = group_df.sort_values(by='date')

        prev_holdings = {}
        prev_date = None

        writelog('='*5 + filer + '='*5)

        for (filing_date, filing_id), filing_rows in group_df.groupby(['date','filing_id']):
            
            writelog(f"\nFiling date: {filing_date}  (Filing ID: {filing_id})")
            current_holdings = {}

            for _, row in filing_rows.iterrows():
                issuer = row['nameOfIssuer']
                shares = row['shares']
                current_holdings[issuer] = shares

            if prev_date is not None:
                for issuer, curShares in current_holdings.items():
                    oldShares = prev_holdings.get(issuer, 0)

                    if issuer not in prev_holdings:
                        writelog(f"  BOUGHT NEW {curShares} shares of {issuer}")
                        X, y = getModelData(issuer, filing_date, filer, 1, curShares, X, y)

                    elif curShares > oldShares:
                        diff = curShares - oldShares
                        writelog(f"  BOUGHT MORE {diff} shares of {issuer}")
                        X, y = getModelData(issuer, filing_date, filer, curShares/oldShares, diff, X, y)

                    elif curShares < oldShares:
                        diff = curShares - oldShares
                        writelog(f"  SOLD SOME {-diff} shares of {issuer}")
                        X, y = getModelData(issuer, filing_date, filer, curShares/oldShares, diff, X, y)

                for issuer, oldShares in prev_holdings.items():
                    if issuer not in current_holdings:
                        writelog(f"  SOLD ALL {oldShares} shares of {issuer}")
                        X, y = getModelData(issuer, filing_date, filer, -1, -oldShares, X, y)

            prev_holdings = current_holdings
            prev_date = filing_date

    saveJson(X, 'DATA/13Ftrain')
    saveJson(y, 'DATA/13Ftest')

def main():
    start_time = time.time()
    data = parseData()
    inferAction(data)
    end_time = time.time()
    writelog("Total execution time:" + str(end_time - start_time))


if __name__ == "__main__":
    main()
