### Refactored feature pipeline: read filings & holdings from DuckDB, write train/test to DuckDB
import os
import time
import json
import threading
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import duckdb
import requests
from bs4 import BeautifulSoup
import yfinance.utils as yfu
import pandas_market_calendars as mcal
from tqdm import tqdm

from getCompanyInfo import getStockInfo

# --- Configuration ---
SOURCE_DB = 'DATA/sec_filings.duckdb'      # existing filings+holdings
FEATURE_DB = 'DATA/sec_features.duckdb'    # new DB for train/test
MAX_WORKERS = min(os.cpu_count(), 16)

# locks for thread-safe writes
_DB_LOCK = threading.Lock()

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
        metrics JSON
    )""")
    conn.execute("""
    CREATE TABLE test_features (
        filer TEXT,
        filing_date TIMESTAMP,
        holding TEXT,
        metrics JSON
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
        ORDER BY f.filer_name, f.date
    """).df()
    conn.close()
    # group into list of (filer, sub_df)
    grouped = [(f, g.drop(columns=['filer_name']).reset_index(drop=True))
               for f, g in df.groupby('filer_name')]
    return grouped


def _process_single_filer(filer, df_group):
    '''
        Iterates through filings for a single filer, extracting train/test features.
         - Skips the first filing (no previous data to compare)
         - For each subsequent filing, compares holdings to the previous one
         - Detemines if a holding was added, removed, or changed
         - For each change, retrieves stock info and prepares train/test data  
    '''
    X_rows, y_rows = [], []
    prev = {}  # mapping holding->(shares,value)
    for _, row in df_group.iterrows():
        date = row['filing_date']
        holding = row['holding']; shares = row['shares']; value = row['value']
        # first filing: init prev
        if not prev:
            prev = {holding:(shares,value)}
            continue

        # compare to prev holdings
        for h,(old_shares,old_val) in list(prev.items()):
            if h not in df_group['holding'].values:
                # full liquidation
                prcnt, dsh, dv = -1, -old_shares, -old_val
                train, test = getStockInfo(h, date, filer, prcnt, dsh, dv)
                if train and test:
                    X_rows.append((filer,date,h,json.dumps(train)))
                    y_rows.append((filer,date,h,json.dumps(test)))

        cur = {row['holding']:(row['shares'],row['value']) for _,row in df_group[df_group['filing_date']==date].iterrows()}
        for h,(cur_sh,cur_val) in cur.items():
            old_sh,old_val = prev.get(h,(0,0))
            if old_sh==0: # bought new holding
                prcnt, dsh, dv = 1, cur_sh, cur_val-old_val
            elif cur_sh>old_sh: # bought some more
                prcnt, dsh, dv = cur_sh/old_sh, cur_sh-old_sh, cur_val-old_val
            elif cur_sh<old_sh: # sold some but not all
                prcnt, dsh, dv = (cur_sh/old_sh if old_sh else 1), cur_sh-old_sh, cur_val-old_val
            else:
                continue
            train, test = getStockInfo(h, date, filer, prcnt, dsh, dv)
            if train and test:
                X_rows.append((filer,date,h,json.dumps(train)))
                y_rows.append((filer,date,h,json.dumps(test)))
        prev = cur
    return X_rows, y_rows

# --- Insert rows into feature DB ---
def insert_rows(table, rows):
    if not rows: return
    with _DB_LOCK:
        conn = duckdb.connect(FEATURE_DB)
        conn.executemany(f"INSERT INTO {table} VALUES (?, ?, ?, ?)", rows)
        conn.close()

# --- Main orchestration ---
def main():
    start = time.time()
    init_feature_db()
    grouped = load_grouped_filings()
    print(f"Found {len(grouped)} filers to process")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_process_single_filer, f, df): f for f,df in grouped}
        for fut in tqdm(as_completed(futures)):
            X_rows, y_rows = fut.result()
            insert_rows('train_features', X_rows)
            insert_rows('test_features', y_rows)
    duration = time.time()-start
    print(f"Completed in {duration:.1f}s")

if __name__ == '__main__':
    main()
