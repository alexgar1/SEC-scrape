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
import json


TEST = True

if not TEST:
    os.system('rm DATA/holdings.feather')
    os.system('rm DATA/filings.feather')
    FILINGDB = 'DATA/filings.feather'
    HOLDINGDB = 'DATA/holdings.feather'

else:
    os.system('rm DATA/TESTholdings.feather')
    os.system('rm DATA/TESTfilings.feather')
    FILINGDB = 'DATA/TESTfilings.feather'
    HOLDINGDB = 'DATA/TESTholdings.feather'

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
CIKFILE = 'companyCIK.json'
NVIDIA = ('NVIDIA','Archives/edgar/data/1045810/')
AMAZON = ('AMAZON','Archives/edgar/data/1018724')
TESLA = ('TESLA','Archives/edgar/data/1318605')
APPLE = ('APPLE','Archives/edgar/data/320193') ### 
MICROSOFT = ('MICROSOFT','Archives/edgar/data/789019')
SCION = ('SCION ASSET MANAGEMENT, LLC','Archives/edgar/data/1649339')
BERKLEY = ('BERKLEY, INC', 'Archives/edgar/data/2051965')


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
# TODO: modify patter to not depend on filename but previous cell entitled 'information table'
pattern_infotable = re.compile(
    r'<td.*?><a href="([^"]+)".*?</a></td>\s*<td.*?>INFORMATION TABLE</td>',
    re.IGNORECASE
)

def saveToFeather(df, tablename):    
    if os.path.exists(tablename):
        try:
            existing_df = pd.read_feather(tablename)
            df = pd.concat([existing_df, df], ignore_index=True)
        except Exception as e:
            print('Error reading or saving to feather table', e)
            
    df.to_feather(tablename)
    print('Saved to', tablename)

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
    print(df)

    if df.empty:
        print("No valid holdings extracted.")
        return False

    saveToFeather(df, HOLDINGDB)

    return True

def getCompany13F(name, cik, quant=3, timeout=30):
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
    succ = 1
    
    try:
        # Start the overall timer
        overall_start = time.time()
        
        # Fetch the company filing directory
        r = requests.get(URL + 'Archives/edgar/data/'+cik, headers=headers, timeout=10)  # Added timeout for initial request
        r.raise_for_status()
        main_text = r.text
        
        folder_links = pattern_folder.findall(main_text)
        if not folder_links:
            print(f"No folder links found for {name}")
            return

        print('='*5, name, '='*5)
        
        for filing, date in folder_links:
            if time.time() - overall_start > timeout:
                print(f"Timeout reached after {timeout} seconds, moving to next filer")
                break
                
            if succ > quant:
                break
            
            try:
                # Fetch the filing page with timeout
                filing_url = URL + filing
                r2 = requests.get(filing_url, headers=headers, timeout=10)
                r2.raise_for_status()
                filing_text = r2.text

                # Fetch filing detail with timeout
                match_index = pattern_index.search(filing_text)
                if not match_index:
                    continue
                    
                detail_href = match_index.group(1)
                detail_url = URL + detail_href
                r3 = requests.get(detail_url, headers=headers, timeout=10)
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
                    print("No valid holdings extracted.")
                    continue
                
                # Get href to holding info
                match_info = pattern_infotable.search(detail_text)
                if not match_info:
                    continue
                    
                info_table_href = match_info.group(1)

                # Scrape holdings with timeout
                info = requests.get(URL + info_table_href, headers=headers, timeout=10).text
                print(name, date)
                
                if extractInvestments(info, filingid):
                    saveToFeather(filingdf, FILINGDB)

                succ += 1
                filing_time = time.time() - overall_start
                print(f"Filing scrape time: {filing_time:.2f} seconds")
                overall_start = time.time()
                
            except requests.exceptions.Timeout:
                print(f"Timeout occurred while processing filing {filing}, skipping")
                continue
            except requests.exceptions.RequestException as e:
                print(f"Request error occurred: {e}, skipping this filing")
                continue
            except Exception as e:
                print(f"Unexpected error processing filing: {e}, skipping this filing")
                continue

    except requests.exceptions.Timeout:
        print(f"Initial request for {name} timed out, moving to next filer")
        return
    except Exception as e:
        print(f"Error during scraping 13F filings for {name}:\n{e}")
        return




def main():
    start_time = time.time()
    with open(CIKFILE,'r') as f:
        ciks = json.load(f)

    for i, key in enumerate(ciks):
        if i < 2:
            company_info = ciks[key]
            getCompany13F(company_info['title'], str(company_info['cik_str']), 2)
            print('Saved', i, 'companies')

    end_time = time.time()
    print("Total execution time:", end_time - start_time)

if __name__ == "__main__":
    main()
