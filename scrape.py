import re, requests, time


def extract_first_folder_link():
    url = "https://www.sec.gov/Archives/edgar/data/1045810/"
    headers = {"User-Agent": "MyCompany (myemail@company.com)"}
    
    # Fetch the page
    r = requests.get(url, headers=headers)
    # Ensure valid response
    r.raise_for_status()


    # regex find href
    #- >\s*<img[^>]*src="/icons/folder.gif" : identifies first folder
    pattern = re.compile(r'<a\s+href="([^"]+)"[^>]*>\s*<img[^>]*src="/icons/folder.gif"')
    match = pattern.search(r.text)
    
    if match:
        # Return the captured href
        return match.group(1)
    else:
        return None

if __name__ == "__main__":
    start = time.time()
    link = extract_first_folder_link()
    end = time.time()

    print('Execution time', end-start)

    if link:
        print("First folder link found:", link)
    else:
        print("No folder link found.")
