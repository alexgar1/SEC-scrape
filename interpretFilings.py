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

os.system('rm DATA/13Ftrain.feather')
os.system('rm DATA/13Ftest.feather')

# TODO
# Include open price day of filing in training data
# Figure out data pipeline for training data

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "max-age=0",
    "Priority": "u=0, i",
    "Referer": "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000013/0001045810-25-000013-index.html",
    "Sec-Ch-Ua": '"Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": "macOS",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
}
PERIOD = 3 # number days to look back and forward for stock data

def saveToFeather(df, tablename):
    if os.path.exists(tablename):
        try:
            existing_df = pd.read_feather(tablename)
            df = pd.concat([existing_df, df], ignore_index=True)
        except Exception as e:
            print('Error reading or saving to feather table', e)

    # Convert only non-list, non-dict, and non-int columns to string type
    for col in df.columns:
        if not df[col].apply(lambda x: isinstance(x, (list, dict, int))).all():
            df[col] = df[col].astype(str)
    
    df.to_feather(tablename)
    print('Saved to ' + tablename)


def getTicker(company_name):
    # url = f"https://query2.finance.yahoo.com/v1/finance/search?q={company_name}"
    # response = requests.get(url, headers=HEADERS)
    # if response.status_code == 200:
    #     data = response.json()
    #     if "quotes" in data and len(data["quotes"]) > 0:
    #         return data["quotes"][0]["symbol"]
        
    yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}

    res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
    data = res.json()

    if "quotes" in data and len(data["quotes"]) > 0:
        return data['quotes'][0]['symbol']
    
    print(f'ERROR: Ticker for {company_name} not found')
    return None


def getOpenDays(date):
    # Get price info for the marker day before and after the filing date
    date = date.tz_localize(None)
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=date - pd.Timedelta(days=7), end_date=date + pd.Timedelta(days=7))
    last_market_day = schedule[schedule.index < date].index[-1-PERIOD]
    next_market_day = schedule[schedule.index > date].index[PERIOD]
    return last_market_day, next_market_day

def getNumShares(start, ticker_info):
    df = ticker_info.quarterly_income_stmt
    if df.empty:
        print("No quarterly income statement data available.")
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
            # This column is "on or before" the date we want
            valid_cols.append((col, cdate))
    if not valid_cols:
        # Means no columns had a date <= start
        print("No quarterly data columns are on or before", start_date)
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

    print('No valid number of shares found.')
    return None



def getPriceInfo(start, end, ticker, train, ticker_info=None):
    # get stock price data
    data = yf.download(ticker, start=start, end=end, interval="1h", auto_adjust=True)
    if data.empty:
        print(f"Error: No stock data found for {ticker} for the given period {start} to {end}")
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
    volume = data['Volume'].values.reshape(-1,1)
    print(data['Close'])


    if train:
        realprice = [float(i[0] * shares) for i in data['Close'].values.reshape(-1,1)]
        realpriceLs = realprice[-PERIOD*7:]
        volumeLs = volume[-PERIOD*7:]
        print('ååååæ',volumeLs)
        priceInfo = realpriceLs, volumeLs, data['Close'].iloc[-1]
        return priceInfo
    else:
        realprice = [float(i[0] * shares) for i in data['Open'].values.reshape(-1,1)]
        train = {
            'RealPrice': realprice[:PERIOD*7], 
            'Volume': volume[:PERIOD*7]}
        return train



def getStockInfo(issuer, filingdate, filer, prcntChng, diffShares):
    ''' 
    Gets stock info for testing a traiing for a given issuer
        TRAIN: stock info and prices before announcement
        TEST: stock RETURNS after announcment
    '''
    ticker = getTicker(issuer)
    if not ticker:
        return None, None
    
    ticker_info = yf.Ticker(ticker)
    
    # Download data
    before, after = getOpenDays(filingdate)
    print(before, filingdate)
    print('holding')
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

    # investment info
    trainInfo = {} 

    # Convert keys in holdingPriceTrainData to strings
    trainInfo['holdingPriceTrainData'] = holdingPriceTrainData
    trainInfo['holdingVolume'] = holdingVolume
    trainInfo['EBITDA'] = ticker_info.info.get('ebitda', None)
    trainInfo['Debt-Cash'] = debt - cash if debt is not None and cash is not None else None
    print('benchmark')
    trainInfo['Benchmark'] = getPriceInfo(before, filingdate, '^GSPC', train=True)

    # investor info
    filerTicker = getTicker(filer)
    print(filerTicker)
    filerInfo = getPriceInfo(before, filingdate, filerTicker, train=True, ticker_info=yf.Ticker(filerTicker))
    if filerInfo is None:
        return None, None
    if filerInfo[0] is None or filerInfo[1] is None:
        print(f"Error: No stock data found for filer {filerTicker} for the given period {before} to {filingdate}")
        return None, None
    trainInfo['filerPrices'] = filerInfo[0]
    trainInfo['filerVolume'] = filerInfo[1]
    trainInfo['percentChng'] = prcntChng
    trainInfo['diffShares'] = diffShares # how many more or less shares (negative if less)
    trainInfo['diffDollars'] = diffShares * lastprice # " in dollars

    testInfo = getPriceInfo(filingdate, after, ticker, train=False, ticker_info=ticker_info)
    if testInfo is None:
        print(f"Error: No stock data found for {ticker} for the given period {filingdate} to {after}")
        return None, None

    
    return trainInfo, testInfo


def parseData():
    try:
        df_filings = pd.read_feather('DATA/filings.feather')
        df_holdings = pd.read_feather('DATA/holdings.feather')
    except Exception as e:
        print("Error reading .feather files:", e)
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
    train, test = getStockInfo(issuer, filing_date, filer, prcnt, diff)
    if train is not None and test is not None:
        X[(filer, issuer, filing_date)] = train
        y[(filer, issuer, filing_date)] = test


def inferAction(grouped):
    X = {}
    y = {}

    for filer, group_df in grouped:
        print(filer)
        group_df = group_df.sort_values(by='date')

        prev_holdings = {}
        prev_date = None

        print('='*5,filer,'='*5)

        for (filing_date, filing_id), filing_rows in group_df.groupby(['date','filing_id']):
            print(f"\nFiling date: {filing_date}  (Filing ID: {filing_id})")
            current_holdings = {}

            for _, row in filing_rows.iterrows():
                issuer = row['nameOfIssuer']
                shares = row['shares']
                current_holdings[issuer] = shares

            if prev_date is not None:
                for issuer, curShares in current_holdings.items():
                    oldShares = prev_holdings.get(issuer, 0)

                    if issuer not in prev_holdings:
                        print(f"  BOUGHT NEW {curShares} shares of {issuer}")
                        getModelData(issuer, filing_date, filer, 1, curShares, X, y)

                    elif curShares > oldShares:
                        diff = curShares - oldShares
                        print(f"  BOUGHT MORE {diff} shares of {issuer}")
                        getModelData(issuer, filing_date, filer, curShares/oldShares, diff, X, y)

                    elif curShares < oldShares:
                        diff = curShares - oldShares
                        print(f"  SOLD SOME {-diff} shares of {issuer}")
                        getModelData(issuer, filing_date, filer, curShares/oldShares, diff, X, y)

                for issuer, oldShares in prev_holdings.items():
                    if issuer not in current_holdings:
                        print(f"  SOLD ALL {oldShares} shares of {issuer}")
                        getModelData(issuer, filing_date, filer, -1, -oldShares, X, y)

            # else:
            #     print("  (No prior filing to compare)")

            prev_holdings = current_holdings
            prev_date = filing_date



    saveToFeather(pd.DataFrame(X), 'DATA/13Ftrain.feather')
    saveToFeather(pd.DataFrame(y), 'DATA/13Ftest.feather')

def main():
    start_time = time.time()
    data = parseData()
    inferAction(data)
    end_time = time.time()
    print("Total execution time:", end_time - start_time)


if __name__ == "__main__":
    main()
