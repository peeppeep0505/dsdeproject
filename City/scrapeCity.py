import os
import json
import csv

# Specify the folder containing the JSON files
folder_path = "2018"
csv_file_path = 'City/' + folder_path + '_city.csv'

counter = 0
data_rows = []

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    
    # if counter == 10: break
    file_path = os.path.join(folder_path, filename)
    
    # Open and read the JSON file
    with open(file_path, "r", encoding="utf-8") as json_file:
        try:
            data = json.load(json_file)
            data = data['abstracts-retrieval-response']['item']['bibrecord']['head']['author-group']

            print(counter)
            counter += 1

            if isinstance(data, list):
                for affiliation in data:
                    result = {}

                    # Get city name
                    city = ''
                    if affiliation.get('affiliation'):
                        if affiliation.get('affiliation').get('city'):
                            city = affiliation['affiliation'].get('city')
                        else:
                            continue
                    else:
                        continue
                    
                    # Get only city
                    if "," in city:
                        city = city.split(', ')[-1]

                    result['city'] = city

                    # Get numbers of author
                    if affiliation.get('author'):
                        result['author_count'] = len(affiliation['author'])
                    else:
                        continue
                    
                    data_rows.append(result)

            elif isinstance(data, dict):
                result = {}

                # Get city name
                city = ''
                if data['affiliation'].get('city'):
                    city = data['affiliation'].get('city')
                else:
                    continue
                
                # Get only city
                if "," in city:
                    city = city.split(', ')[-1]

                result['city'] = city

                # Get numbers of author
                if data.get('author'):
                    result['author_count'] = len(data['author'])
                else:
                    continue

                data_rows.append(result)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file {filename}: {e}")

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