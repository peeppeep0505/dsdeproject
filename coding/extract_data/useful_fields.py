import os
import json
import csv

# Specify the folder containing the JSON files
folder_path = "2023"
csv_file_path = 'test2023.csv'
# counter = 0
data_rows = []

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    
    # if counter == 10: break
    file_path = os.path.join(folder_path, filename)
    
    # Open and read the JSON file
    with open(file_path, "r", encoding="utf-8") as json_file:
        try:
            data = json.load(json_file)
            data = data['abstracts-retrieval-response']

            result = {}


            # title
            result['title'] = data['item']['bibrecord']['head']['citation-title']


            # authors
            result['authors'] = []

            author_json = data['authors']['author']

            if type(author_json) == list:
                for author in author_json:
                    result['authors'].append(author['ce:indexed-name'])
            else:
                result['authors'].append(author_json['ce:indexed-name'])


            # affiliations
            result['affiliations'] = []

            affiliation_json = data['affiliation']

            if type(affiliation_json) == list:
                for affiliation in affiliation_json:
                    result['affiliations'].append(affiliation['affilname'])
            else:
                result['affiliations'].append(affiliation_json['affilname'])

            #citedby-count
            result['citedby'] = data["coredata"]['citedby-count']


            # mainterms
            result['mainterms'] = []

            idxterms_json = data['idxterms']
            
            # Check if null
            if idxterms_json:
                if type(idxterms_json['mainterm']) == list:
                    for mainterm in idxterms_json['mainterm']:
                        result['mainterms'].append(mainterm['$'])
                else:
                    result['mainterms'].append(idxterms_json['mainterm']['$'])


            # subject-areas
            result['subject_areas'] = []

            for subject in data['subject-areas']['subject-area']:
                result['subject_areas'].append(subject['$'])


            # publisher
            source_json = data['item']['bibrecord']['head']['source']
            
            result['publisher'] = source_json.get('publisher', {}).get('publishername', None)
            
            # Many missing data
            # public date
            # publication_date_json = source_json.get('publicationdate', {})
            # try:
            #     year = int(publication_date_json.get('year'))
            #     month = int(publication_date_json.get('month'))
            #     day = int(publication_date_json.get('day'))
            #     result['publication_date'] = date(year, month, day)
            # except (ValueError, TypeError):  # Catch invalid values or missing fields
            #     result['publication_date'] = None  # Fallback for invalid or incomplete dates

            data_rows.append(result)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file {filename}: {e}")
    
    # counter += 1

if data_rows:
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        # Use the keys from the first dictionary as the header
        fieldnames = data_rows[0].keys()

        # Create a CSV writer
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header
        writer.writeheader()

        # Write all rows
        writer.writerows(data_rows)

    print(f"CSV file '{csv_file_path}' has been created.")
else:
    print("No data to write to CSV.")