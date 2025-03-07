import pandas as pd
import sys
import datetime, time
import yfinance as yf
import requests
import pandas_market_calendars as mcal
import os
from sklearn.linear_model import LinearRegression
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



def saveToFeather(df, tablename):    
    if os.path.exists(tablename):
        try:
            existing_df = pd.read_feather(tablename)
            df = pd.concat([existing_df, df], ignore_index=True)
        except Exception as e:
            print('Error reading or saving to feather table', e)
            
    df.to_feather(tablename)
    print('Saved to'+tablename)


def getTicker(company_name):
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={company_name}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        if "quotes" in data and len(data["quotes"]) > 0:
            return data["quotes"][0]["symbol"]
    print(f'ERROR: Ticker for {company_name} not found')
    return None

def getReturn(issuer, start, relative_to='Close'):
    ticker = getTicker(issuer)
    if not ticker:
        return None

    def getOpenDays(date):
        # Get price info for the marker day before and after the filing date
        date = date.tz_localize(None)
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=date - pd.Timedelta(days=10), end_date=date + pd.Timedelta(days=10))
        last_market_day = schedule[schedule.index < date].index[-2]
        next_market_day = schedule[schedule.index > date].index[1]
        return last_market_day, next_market_day
    
    def getBeta(data, start, end):
        # get benchmark data from total market index for beta value
        benchData = yf.download('^GSPC', start=start, end=end, interval="1h", auto_adjust=True)
        if benchData.empty:
            print(f"Error: No benchmark data found for the given period {start} to {end}")
            return None

        # convert timezone
        benchData.index = benchData.index.tz_convert('US/Eastern')
        data.index = data.index.tz_convert('US/Eastern')

        benchData['Percent Return'] = (benchData[relative_to] / benchData[relative_to].iloc[0] - 1) * 100

        alignedData = pd.concat([data['Percent Return'], benchData['Percent Return']], axis=1).dropna()
        alignedData.columns = ['Stock', 'Benchmark']
        X = alignedData['Benchmark'].values.reshape(-1, 1)
        y = alignedData['Stock'].values
        model = LinearRegression().fit(X, y)
        beta = model.coef_[0]

        return beta
    
    def getPriceInfo(start, end, ticker, training=False):
        # get stock price data
        data = yf.download(ticker, start=start, end=end, interval="1h", auto_adjust=True)
        if data.empty:
            print(f"Error: No stock data found for {ticker} for the given period {start} to {end}")
            return None
        
        # Calculate the return relative to the specified variable
        data['Percent Return'] = (data[relative_to] / data[relative_to].iloc[0] - 1) * 100

        output = data[['Close', 'Percent Return']].copy()

        if training:
            # Fetch additional financial metrics
            ticker_info = yf.Ticker(ticker)
            debt = ticker_info.info.get('totalDebt', None)
            cash = ticker_info.info.get('totalCash', None)

            shares_outstanding = ticker_info.info.get('sharesOutstanding', None)
            output['MarketCap'] = shares_outstanding * data['Close'].iloc[-1] if shares_outstanding is not None else None
            output['EBITDA'] = ticker_info.info.get('ebitda', None)
            output['Debt-Cash'] = debt - cash if debt is not None and cash is not None else None
            output['Beta'] = getBeta(data, start, end)

        return output

    # Download data
    before, after = getOpenDays(start)
    pricedata_train = getPriceInfo(before, start, ticker, training=True)
    pricedata_test = getPriceInfo(start, after, ticker)
    
    print('BEFORE')
    print(pricedata_train)
    saveToFeather(pricedata_train, 'DATA/13Ftrain.feather')

    
    print('AFTER')
    print(pricedata_test)
    saveToFeather(pricedata_test, 'DATA/13Ftest.feather')
    
    return pricedata_train, pricedata_test

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

def inferAction(grouped):
    ''' '''

    for filer, group_df in grouped:
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
                for issuer, curr_shares in current_holdings.items():
                    old_shares = prev_holdings.get(issuer, 0)
                    action = None

                    if issuer not in prev_holdings:
                        action = "BOUGHT NEW"
                        print(f"  BOUGHT NEW {curr_shares} shares of {issuer}")
                        returns = getReturn(issuer, filing_date)

                    elif curr_shares > old_shares:
                        diff = curr_shares - old_shares
                        action = "BOUGHT MORE"
                        print(f"  BOUGHT MORE {diff} shares of {issuer}")
                        returns = getReturn(issuer, filing_date)

                    elif curr_shares < old_shares:
                        diff = old_shares - curr_shares
                        if curr_shares == 0:
                            action = "SOLD ALL"
                            print(f"  SOLD ALL {old_shares} shares of {issuer}")
                        else:
                            action = "SOLD SOME"
                            print(f"  SOLD SOME {diff} shares of {issuer}")

                        returns = getReturn(issuer, filing_date)

                for issuer, old_shares in prev_holdings.items():
                    if issuer not in current_holdings:
                        print(f"  SOLD ALL {old_shares} shares of {issuer}")
                        returns = getReturn(issuer, filing_date)
            else:
                print("  (No prior filing to compare)")

            prev_holdings = current_holdings
            prev_date = filing_date

def main():
    start_time = time.time()
    data = parseData()
    inferAction(data)
    end_time = time.time()
    print("Total execution time:", end_time - start_time)

if __name__ == "__main__":
    main()
