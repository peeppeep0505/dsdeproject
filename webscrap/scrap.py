import requests
import csv

API_KEY = "90a48ae5e4a541c796c28ef58d3c9c75"
BASE_URL = "https://api.elsevier.com/content/search/scopus"

TOTAL_RESULTS = 1000 
PAGE_SIZE = 25 
start = 4000

results_count = 0  

with open('plsplspls.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["title", "authors", "affiliations", "citedby", "publisher", "Year"])  

    while results_count < TOTAL_RESULTS:
        params = {
            "query": "AF-ID(60028190)",  
            "count": PAGE_SIZE, 
            "start": start,  
            "sort": "plf-f", 
            "yearFrom": "1999",  
            "yearTo": "2018", 
            "apikey": API_KEY,  
        }
        response = requests.get(BASE_URL, params=params)

        if response.status_code == 200:
            data = response.json()
            entries = data.get("search-results", {}).get("entry", [])
            total_results = int(data.get("search-results", {}).get("opensearch:totalResults", 0))

            print(f"Total Results available: {total_results}")

            if not entries:
                print("No results found.")
                break

            for entry in entries:
                title = entry.get("dc:title", "No Title")
                authors = entry.get("dc:creator", [])
                if isinstance(authors, list):
                    authors = ', '.join(authors) if authors else "No Author"
                scopus_id = entry.get("dc:identifier", "").replace("SCOPUS_ID:", "")
                citedby_count = entry.get("citedby-count", "No citation count")
                cited_by_count = entry.get("citedby-count", "No Cited By Count")

                affiliations = entry.get("affiliation", [])
                if affiliations:
                    affiliation_list = [
                        f"{affil.get('affilname', 'No Affiliation')} ({affil.get('affiliation-country', 'No Country')})"
                        for affil in affiliations
                    ]
                    affiliation_str = "; ".join(affiliation_list)
                else:
                    affiliation_str = "No Affiliation"

                cover_date = entry.get("prism:coverDate", "")
                year = cover_date.split("-")[0] if cover_date else "No Year"

                publisher = "No Publisher"

                if scopus_id:
                    abstract_url = f"https://api.elsevier.com/content/abstract/scopus_id/{scopus_id}"

                    headers = {
                        "Accept": "application/json",
                        "X-ELS-APIKey": API_KEY
                    }
                    response2 = requests.get(abstract_url, headers=headers)

                    if response2.status_code == 200:
                        data2 = response2.json()
                        publisher = data2.get("abstracts-retrieval-response", {}).get("coredata", {}).get("dc:publisher", "No Publisher")

                writer.writerow([title, authors, affiliation_str, citedby_count, publisher, year])

                results_count += 1

                if results_count >= TOTAL_RESULTS:
                    break

            start += PAGE_SIZE

            if results_count >= total_results:  
                print(f"Reached the total results: {results_count}")
                break

        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
            break

print(f"CSV file 'plsplspls.csv' has been created successfully with {results_count} results.")