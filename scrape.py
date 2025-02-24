### Written by Alex Garrett alexgarrett2468[at]gmail.com 2025

import re, requests, time, os
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf


URL = "https://www.sec.gov/"
NVIDIA = ('NVIDIA','Archives/edgar/data/1045810/')
AMAZON = ('AMAZON','Archives/edgar/data/1018724')

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
# pattern_info_table = re.compile(
#     r'<a\s+href="([^"]+information_table\.xml)"',
#     re.IGNORECASE
# )

def saveToFeather(df, tablename):    
    if os.path.exists(tablename):
        try:
            existing_df = pd.read_feather(tablename)
            df = pd.concat([existing_df, df], ignore_index=True)
        except Exception as e:
            print('Error reading or saving to feather table', e)
            
    df.to_feather(tablename)
    print('Saved to',tablename)

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
        
        # We expect at least 7 columns for a valid data row
        if len(columns) < 7:
            continue
        
        try:
            # Pull text from each cell, stripping whitespace and unwanted chars
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
            # This typically indicates a header or non-numeric row; skip it
            continue

    df = pd.DataFrame(holdings)
    print(df)

    if df.empty:
        print("No valid holdings extracted.")
        return

    saveToFeather(df, 'holdings.feather')

    return 

def getCompany13F(company, quant=3):
    """
    This function scrapes the SEC website for 13F filings from a given company
    and retrieves asset information.

    1. Fetch the given basepath from the SEC site, find all folder links,
    2. For each folder link navigate to the filing details page,
    3. Locate the information table with security holdings table,
    4. Record all holdings and parse their information.
    """

    headers = {"User-Agent": "MyCompany (myemail@company.com)"}

    r = requests.get(URL + company[1], headers=headers) # company filing dir
    r.raise_for_status()  # Raises an HTTPError if the response was an error
    main_text = r.text
    filings = []
    succ = 0
    try:
        folder_links = pattern_folder.findall(main_text)

        start_time = time.time()
        for filing, date in folder_links:
            if succ > quant:
                break

            # Fetch the filing page.
            filing_url = URL + filing
            r2 = requests.get(filing_url, headers=headers)
            r2.raise_for_status()
            filing_text = r2.text

            # Fetch filing detail
            match_index = pattern_index.search(filing_text)
            if not match_index:
                continue
            detail_href = match_index.group(1)
            detail_url = URL + detail_href
            r3 = requests.get(detail_url, headers=headers)
            r3.raise_for_status()
            detail_text = r3.text

            # Find Accession Number (filing_id)
            match_filingid = pattern_filingid.search(detail_text)
            filingid = match_filingid.group(1)
            filings.append({
                'filing_id': filingid,
                'cik': filingid[:10],
                'filer_name': company[0], # company name
                'date': date
                })
            
            # TODO: modify patter to not depend on filename but previous cell entitled 'information table'
            match_info = pattern_infotable.search(detail_text)
            if not match_info:
                continue
            info_table_href = match_info.group(1)
            print('ASDF',info_table_href)

            # Scrape holdings
            info = requests.get(URL + info_table_href, headers=headers).text
            print(company[0], date)
            extractInvestments(info, filingid)
            succ += 1
            end_time = time.time()
            print("Filing scrape time:", end_time - start_time)
            start_time = time.time()

            # break # this only gets the latest filing; remove to get all filings

    except Exception as e:
        print("Error during scraping 13F filings", e)

    df = pd.DataFrame(filings)

    if df.empty:
        print("No valid holdings extracted.")
        return
    
    saveToFeather(df, 'filings.feather')


def main():
    start_time = time.time()
    getCompany13F(NVIDIA)
    end_time = time.time()
    print("Total execution time:", end_time - start_time)

if __name__ == "__main__":
    main()
