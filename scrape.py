### Written by Alex Garrett alexgarrett2468[at]gmail.com 2025

# This script scrapes the sec edgar database for 13f filings without using the API (the free API caps requests at 100)
# 1. Accesses company filings archive
# 2. Searches for 13F filings
# 3. Extracts company holdings and saves them to a table

# TODO
# Implement scraping for 8k and 10k filings.
# Extract financial information and perform NLP for sentiment analysis

import re, requests, time, os
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
import duckdb
import json
from tqdm import tqdm
import logging
import threading
import itertools
import random
import requests, threading



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s %(message)s",
    handlers=[
        logging.FileHandler("LOGS/scrape.txt", mode="w")
    ]
)
logger = logging.getLogger(__name__)

DB_FILE = 'DATA/sec_filings.duckdb'


HEADERS = {
    "User-Agent": "Alex Garrett alexgarrett2468@gmail.com",
}

URL = "https://www.sec.gov/"
CIKFILE = 'companyCIK.json'

pattern_folder = re.compile(
    r'<td><a href="(?P<href>/Archives/edgar/data/\d+/\d+)".*?</a></td><td></td><td>(?P<date>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})</td>',
    re.IGNORECASE
)
pattern_dates = re.compile(
    r'<td>(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})</td>',
    re.IGNORECASE
)
pattern_dates = re.compile(
    r'<a\s+href="([^"]+)"[^>]*>\s*<img[^>]*src="/icons/folder.gif"',
    re.IGNORECASE
)
pattern_index = re.compile(
    r'<a\s+href="([^"]+-index\.html)"',
    re.IGNORECASE
)
pattern_date = re.compile(
    r'</a></td><td></td><td>({^<+})</td></tr>',
    re.IGNORECASE
)
pattern_filingid = re.compile( # actually called the 'Acession Number'
    r'(\d{10}-\d{2}-\d{6})',
    re.IGNORECASE
)
# TODO: modify pattern to not depend on filename but previous cell entitled 'information table'
pattern_infotable = re.compile(
    r'<td.*?><a href="([^"]+)">\w+\.html.*?</a></td>\s*<td.*?>INFORMATION TABLE</td>',
    re.IGNORECASE
)

# connect and initialize DuckDB
conn = duckdb.connect(DB_FILE)
conn.execute("""
CREATE TABLE IF NOT EXISTS filings (
    filing_id VARCHAR PRIMARY KEY,
    filing_type VARCHAR,
    cik VARCHAR,
    filer_name VARCHAR,
    date TIMESTAMP
)
""")
conn.execute("""
CREATE TABLE IF NOT EXISTS holdings (
    id VARCHAR,
    nameOfIssuer VARCHAR,
    titleOfClass VARCHAR,
    cusip VARCHAR,
    value BIGINT,
    sshPrnamt BIGINT,
    putCall VARCHAR,
    PRIMARY KEY (id, cusip),
    FOREIGN KEY(id) REFERENCES filings(filing_id)
    
)
""")

def save_filings(df: pd.DataFrame):    
    # upsert into filings
    conn.register('tmp_filings', df)
    conn.execute('''
    INSERT INTO filings SELECT * FROM tmp_filings
    ON CONFLICT (filing_id) DO UPDATE SET
      filing_type = EXCLUDED.filing_type,
      cik = EXCLUDED.cik,
      filer_name = EXCLUDED.filer_name,
      date = EXCLUDED.date;
    ''' )
    conn.unregister('tmp_filings')
    logger.info('Filings saved to DuckDB')

def save_holdings(df: pd.DataFrame):
    # normalize nested column
    df = df.assign(sshPrnamt=df['shrsOrPrnAmt'].apply(lambda x: x.get('sshPrnamt', None)))
    df = df.drop(columns=['shrsOrPrnAmt'])
    conn.register('tmp_holdings', df)
    conn.execute('''
    INSERT INTO holdings (id, nameOfIssuer, titleOfClass, cusip, value, sshPrnamt, putCall)
    SELECT id, nameOfIssuer, titleOfClass, cusip, value, sshPrnamt, putCall FROM tmp_holdings;
    ''')
    conn.unregister('tmp_holdings')
    logger.info('Holdings saved to DuckDB')


def backoffFn(fn, *args, **kwargs):
    """
    Retry fn(*args, **kwargs) on HTTP 401/429, with exponential backoff.
    """
    delays = itertools.chain([1, 2, 4, 8, 16], itertools.repeat(30))
    for delay in delays:
        try:
            resp = fn(*args, **kwargs)
            if hasattr(resp, "status_code") and resp.status_code != 200:
                logger.warning(f"Request failed with status code: {resp.status_code}")
                raise requests.HTTPError(response=resp)
            return resp
        except requests.HTTPError as e:
            code = e.response.status_code if e.response else None
            if code in (401, 429):
                logger.warning(f"Rate‐limited ({code}), retrying in {delay}s…")
                time.sleep(delay)
                continue
            raise
        except requests.RequestException as e:
            # optionally retry on other transient errors
            logger.error(f"Network error {e}, retrying in {delay}s…")
            time.sleep(delay)

def extractInvestments(html_content, filingid):
    """
    Extracts security investment information from the SEC 13F filing HTML.

    Returns:
        list: A list of tuples containing (Company Name, Investment Value).
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    table_rows = soup.find_all('tr')

    holdings = []
    for row in table_rows:
        columns = row.find_all('td')
        
        if len(columns) < 7:
            continue
        
        try:
            name_of_issuer = columns[0].get_text(strip=True).upper()
            title_of_class = columns[1].get_text(strip=True).upper()
            cusip = columns[2].get_text(strip=True)
            
            # columns[3] is sometimes "FIGI" or another data point; skip it if needed
            
            value_str = columns[4].get_text(strip=True).replace(",", "").replace("Â", "")
            shares_str = columns[5].get_text(strip=True).replace(",", "").replace("Â", "")
            put_call = columns[6].get_text(strip=True)
            
            # Convert value and shares into integers; skip row if this fails
            value = int(value_str)
            shares = int(shares_str)
            
            # Build the holding dictionary
            holdings.append({
                'id': filingid,
                'nameOfIssuer': name_of_issuer,
                'titleOfClass': title_of_class,
                'cusip': cusip,
                'value': value,
                'shrsOrPrnAmt': {'sshPrnamt': shares},
                'putCall': put_call
            })
        
            
        except ValueError:
            continue


    df = pd.DataFrame(holdings)
    logger.info(df)

    if df.empty:
        logger.info("No valid holdings extracted.")
        return False

    save_holdings(df)

    return True

def getCompany13F(name, cik, ticker, timeout=60):
    """
    This function scrapes the SEC website for 13F filings from a given company
    and retrieves asset information.

    1. Fetch the given basepath from the SEC site, find all folder links,
    2. For each folder link navigate to the filing details page,
    3. Locate the information table with security holdings table,
    4. Record all holdings and parse their information.
    """

    headers = HEADERS
    filings = []
    succ = 0
    
    try:
        # Start the overall timer
        overall_start = time.time()
        logger.info(f"{'='*5} {name} {'='*5}")
        
        # Fetch the company filing directory
        lurl   = (
            "https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcompany&CIK={cik}"
            "&type=13F-HR&owner=include&count=100"
        )
        logger.info(f"Fetching filings for {name} (CIK: {cik}) at URL: {lurl}")
        r = backoffFn(requests.get, lurl, headers=headers, timeout=10)  # Added timeout for initial request
        r.raise_for_status()
        main_text = r.text


        soup = BeautifulSoup(main_text, 'html.parser')

        for row in soup.find_all('tr')[1:]:
            cols = row.find_all('td')

            if cols and cols[0].get_text(strip=True) == '13F-HR':
                link_tag = cols[1].find('a', id='documentsbutton')
                date_td = cols[3]

                filing_url = URL + link_tag['href']
                date = date_td.get_text(strip=True)

                try:
                    r2 = backoffFn(requests.get, filing_url, headers=headers, timeout=10)
                    r2.raise_for_status()
                    fdetail = r2.text
                    
                    # Find Accession Number (filing_id)
                    match_filingid = pattern_filingid.search(fdetail)
                    filingid = match_filingid.group(1)
                    filinginfo = {
                        'filing_id': filingid,
                        'filing_type': '13F',
                        'cik': filingid[:10],
                        'filer_name': name,
                        'date': pd.to_datetime(date, errors='coerce')
                    }
                    filingdf = pd.DataFrame([filinginfo])
                    if filingdf.empty:
                        logger.info("No valid holdings extracted.")
                        continue

                    match_info = pattern_infotable.search(fdetail)
                    if not match_info:
                        continue
                        
                    hlink = match_info.group(1)

                    # Scrape holdings with timeout
                    logger.info(f"Fetching holdings page {hlink} for {name}")
                    info = backoffFn(requests.get, URL + hlink, headers=headers, timeout=10).text
                    logger.info(f'{name} {date}')
                    
                    conn.begin()
                    try:
                        save_filings(filingdf)          # parent first
                        extractInvestments(info, filingid)  # children second
                        conn.commit()
                    except Exception:
                        conn.rollback()
                        raise

                    succ += 1
                    filing_time = time.time() - overall_start
                    logger.info(f"Filing scrape time: {filing_time:.2f} seconds")
                    overall_start = time.time()

                    
                except requests.exceptions.Timeout:
                    logger.info(f"Timeout occurred while processing filing {filingid}, skipping")
                    continue
                except requests.exceptions.RequestException as e:
                    logger.info(f"Request error occurred: {e}, skipping this filing")
                    continue
                except Exception as e:
                    logger.info(f"Unexpected error processing filing: {e}, skipping this filing")
                    continue

    except requests.exceptions.Timeout:
        logger.info(f"Initial request for {name} timed out, moving to next filer")
        return 0
    except Exception as e:
        logger.info(f"Error during scraping 13F filings for {name}:\n{e}")
        return 0 
    
    return succ




def main():
    start_time = time.time()
    with open(CIKFILE,'r') as f:
        ciks = json.load(f)

    total = 0
    i = 0
    for key in tqdm(ciks):
        company_info = ciks[key]
        succ = getCompany13F(company_info['title'], str(company_info['cik_str']), company_info['ticker'])
        if succ > 0:
            total += succ
            i += 1
            logger.info(f'Saved {total} filings from {i} companies')

    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time}")

if __name__ == "__main__":
    main()