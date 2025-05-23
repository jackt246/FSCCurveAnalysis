import requests
import json
import pandas as pd
import concurrent.futures
import csv

class FileImport:
    def __init__(self):
        self.apiURL = 'https://ebi.ac.uk/emdb/api/'

    def importValidation(self, entry, saveFile=False):
        url = f'{self.apiURL}analysis/{entry}'

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Will raise an error if status code is not 200

            try:
                json_data = response.json()
            except ValueError as e:
                print(f"Failed to parse JSON for entry {entry}: {e}")
                return None

            if saveFile:
                try:
                    with open(f'{entry}_validation.json', 'w') as json_file:
                        json.dump(json_data, json_file)
                except Exception as e:
                    print(f"Could not save file for entry {entry}: {e}")

            print(f"Successfully retrieved data for {entry}")
            return json_data

        except requests.exceptions.RequestException as e:
            print(f"Request error for entry {entry}: {e}")
            return None

class EMDBsearcher():
  def __init__(self):
    self.apiURL = 'https://ebi.ac.uk/emdb/api/'


try:
    df = pd.read_csv('emdb_withhalfmaps.csv')
except FileNotFoundError:
    print("Error: 'emdb_withhalfmaps.csv' not found. Please upload the file to the current working directory.")
except pd.errors.ParserError:
    print("Error: Could not parse the CSV file. Please check the file format.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

EMDlist = df.values.tolist()

fileimporter = FileImport()

fsc_data = []

def process_entry(entry):
    entry_id = entry[0]  # assuming EMDlist is a list of tuples
    json_data = fileimporter.importValidation(entry_id)

    if json_data is None:
        print(f"No data for {entry_id}, skipping.")
        return None

    try:
        data = list(json_data.values())[0]
        fsc = data['fsc']['curves']['fsc']
        return fsc
    except KeyError:
        print(f"No FSC data for {entry_id}, skipping.")
        return None

# Use ThreadPoolExecutor with 20 workers
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
  results = list(executor.map(process_entry, EMDlist))

# Filter out failed (None) results
fsc_data = [fsc for fsc in results if fsc is not None]

print(f"Got {len(fsc_data)} FSC curves")

with open('fsc_curves.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for curve in fsc_data:
        writer.writerow(curve)

print("FSC data saved to fsc_curves.csv")