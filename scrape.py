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
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import threading
import itertools
import random
from sshtunnel import open_tunnel          # pip install sshtunnel
import requests, threading
import os
from dotenv import load_dotenv, find_dotenv
import socks


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s %(message)s",
    handlers=[
        logging.FileHandler("LOGS/scrapeNEW.txt", mode="w")
    ]
)
logger = logging.getLogger(__name__)

DB_FILE = 'DATA/sec_filingsNEW.duckdb'


HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "max-age=0",
    "Priority": "u=0, i",
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

URL = "https://www.sec.gov/"
CIKFILE = 'DATA/companyCIK.json'

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
    r'<td.*?><a href="([^"]+)".*?</a></td>\s*<td.*?>INFORMATION TABLE</td>',
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
            # some libraries don’t raise on 429, so check status
            if hasattr(resp, "status_code") and resp.status_code in (401, 429):
                raise requests.HTTPError(response=resp)
            if resp.status_code != 200:
                logger.warning(f"Request failed with status code: {resp.status_code}")
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

def getCompany13F(name, cik, quant=3, timeout=60):
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
        
        # Fetch the company filing directory
        LATESTURL = URL + f'/edgar/search/#/q=13f-hr&entityName={(10-len(cik))*'0'}{cik}'
        logger.info(f"Fetching filings for {name} (CIK: {cik}) at URL: {LATESTURL}")
        r = backoffFn(requests.get, LATESTURL, headers=headers, timeout=10)  # Added timeout for initial request
        r.raise_for_status()
        main_text = r.text
        
        soup = BeautifulSoup(main_text, "lxml")
        folder_links = [
            a["href"] for a in soup.select('a.preview-file')
            if a.text.startswith("13F-HR")
        ]
        if not folder_links:
            logger.info(f"No folder links found for {name}")
            return 0
        
        logger.info(f"Found {len(folder_links)} folder links for {name}")

        logger.info(f"{'='*5} {name} {'='*5}")
        
        for filing, date in folder_links:
            if time.time() - overall_start > timeout:
                logger.info(f"Timeout reached after {timeout} seconds, moving to next filer")
                break
                
            if succ > quant:
                break
            
            try:
                # Fetch the filing page with timeout
                filing_url = URL + filing
                r2 = backoffFn(requests.get, filing_url, headers=headers, timeout=10)
                r2.raise_for_status()
                filing_text = r2.text

                # Fetch filing detail with timeout
                match_index = pattern_index.search(filing_text)
                if not match_index:
                    continue
                
                detail_href = match_index.group(1)

                detail_url = URL + detail_href
                r3 = backoffFn(requests.get, detail_url, headers=headers, timeout=10)
                r3.raise_for_status()
                detail_text = r3.text

                # Find Accession Number (filing_id)
                match_filingid = pattern_filingid.search(detail_text)
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
                
                # Get href to holding info
                match_info = pattern_infotable.search(detail_text)
                if not match_info:
                    continue
                    
                info_table_href = match_info.group(1)

                # Scrape holdings with timeout
                logger.info(f"Fetching holdings page {info_table_href} for {name}")
                info = backoffFn(requests.get, URL + info_table_href, headers=headers, timeout=10).text
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
                logger.info(f"Timeout occurred while processing filing {filing}, skipping")
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
        succ = getCompany13F(company_info['title'], str(company_info['cik_str']))
        if succ > 0:
            total += succ
            i += 1
            logger.info(f'Saved {total} filings from {i} companies')

    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time}")

if __name__ == "__main__":
    main()