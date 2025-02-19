
from sec_api import QueryApi, RenderApi
import pandas as pd
import os


queryApi = QueryApi(api_key='dbdfe751c552cf417261c5a05fd71cdd046b2b27907dce9fe4891092e1bc33bc')
renderApi = RenderApi(api_key='dbdfe751c552cf417261c5a05fd71cdd046b2b27907dce9fe4891092e1bc33bc')

def flatten_lists(lists):
  return [element for sublist in lists for element in sublist]


# extract .xml files living in documentFormatFiles list
def extract_xml_urls(filing):
  if "documentFormatFiles" in filing and len(filing["documentFormatFiles"]):
    urls = list(map(lambda file: file["documentUrl"], filing["documentFormatFiles"]))
    xml_urls = list(filter(lambda url: url.endswith(".xml") or url.endswith(".XML"), urls))
    return xml_urls
  else:
    return []


def get_all_xml_urls(start, end):
  form_type = "13F"

  lucene_query = 'formType:"{form_type}" AND filedAt:[{start} TO {end}]'.format(form_type=form_type, start=start, end=end)

  xml_urls = []

  query_from = 0

  while query_from < 10:
    query = {
      "query": lucene_query,
      "from": query_from,
      "size": "50",
      "sort": [{ "filedAt": { "order": "desc" } }]
    }

    response = queryApi.get_filings(query)

    if len(response["filings"]) == 0:
      break

    new_xml_urls = list(map(extract_xml_urls, response["filings"]))

    xml_urls = xml_urls + flatten_lists(new_xml_urls)

    query_from += 50
    # break

  return xml_urls


def download_file(url):
  try:
    if "url" in url:
      url = url["url"]

    content = renderApi.get_filing(url)

    url_parts = url.split("/")
    url_numeric_parts = list(filter(lambda part: part.isnumeric(), url_parts))

    # URL: https://www.sec.gov/Archives/edgar/data/1731627/000188852422016521/exh_102.xml
    # CIK: 1731627
    # accession no: 000188852422016521
    # original file name: exh_102.xml
    cik = url_numeric_parts[0]
    accession_no = url_numeric_parts[1]
    original_file_name = url_parts[-1]

    download_dir = "xml-files/" + cik
    file_name = accession_no + "-" + original_file_name
    download_path = download_dir + "/" + file_name

    # create dir if it doesn't exist
    if not os.path.exists(download_dir):
      os.makedirs(download_dir)

    with open(download_path, "w") as f:
      f.write(content)

  except Exception as e:
    print(e)
    print("❌ download failed: {url}".format(url=url))

def main():
    xmls = get_all_xml_urls('2025-01-01', '2025-02-18')
    print('Got', len(xmls), 'filings')
    print(xmls[:5])

if __name__ == '__main__':
    main()