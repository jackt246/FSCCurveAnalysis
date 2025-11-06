import json
import pandas as pd
from emdb.client import EMDB
from tqdm import tqdm
import os

client = EMDB()

#Get entries with half-maps from before 2025
results = client.csv_search('half_map_filename:[* TO *] AND current_status:"REL" AND release_date:[2002-01-01T00:00:00Z TO 2025-06-30T23:59:59Z] AND database:EMDB')
all_ids = results['emdb_id'].astype(str).tolist()
#Prep a list to hold data during the iterations
data_list = []

# Partial CSV file for progressive saving
data_file = "fsc_curves_partial.csv"

# Load partial data if it exists
if os.path.exists(data_file):
    saved_df = pd.read_csv(data_file)
    data_list = saved_df.to_dict(orient="records")
    processed_ids = set(str(i) for i in saved_df["id"].tolist())
    print(f"Loaded {len(saved_df)} previously saved entries")
else:
    data_list = []
    processed_ids = set()

batch = []  # temporary batch to reduce I/O
BATCH_SIZE = 50  # save every 50 processed entries

unprocessed_ids = [i for i in all_ids if i not in processed_ids]
print(f"Prefiltered down to {len(unprocessed_ids)} unprocessed entries")


#populate the df with the entries in results
for emdb_id in tqdm(unprocessed_ids, desc="Processing EMDB entries", unit="entry"):
    try:
        entry = client.get_entry(emdb_id)
        entry_id = entry.id
        entry_method = entry.method

        validation_data = entry.get_validation()
        if not validation_data or not hasattr(validation_data, "plots"):
            continue

        validation_graphs = validation_data.plots
        if not hasattr(validation_graphs, "fsc"):
            continue

        fsc_curves = validation_graphs.fsc

        resolution = getattr(fsc_curves, "resolution", None)
        fsc_corrected = getattr(fsc_curves, "fsc_corrected", None)
        fsc_masked = getattr(fsc_curves, "fsc_masked", None)
        fsc_phaserandom = getattr(fsc_curves, "phaserandomization", None)
        fsc_unmasked = getattr(fsc_curves, "fsc", None)

        entry_data = {
            'id': entry_id,
            'method': entry_method,
            'resolution': resolution,
            'fsc_corrected': fsc_corrected,
            'fsc_phaserandom': fsc_phaserandom,
            'fsc_masked': fsc_masked,
            'fsc_unmasked': fsc_unmasked
        }

        data_list.append(entry_data)
        batch.append(entry_data)

        if len(batch) >= BATCH_SIZE:
            pd.DataFrame(batch).to_csv(
                data_file,
                mode='a',
                header=not os.path.exists(data_file),  # write headers only once
                index=False
            )
            batch.clear()
            print(f"Checkpoint saved ({len(processed_ids)} entries total)")

    except Exception as e:
        print(f"Failed to process {entry.id}: {e}")

if batch:
    pd.DataFrame(batch).to_csv(
        data_file,
        mode='a',
        header=not os.path.exists(data_file),
        index=False
    )

#a df to take the entry information
columns = ['id', 'method', 'resolution', 'fsc_corrected', 'fsc_phaserandom', 'fsc_masked', 'fsc_unmasked']
fsc_df = pd.DataFrame(data_list, columns=columns)

fsc_df.to_csv("../clustering/fsc_curves/fsc_curves_all.csv", index=False)