import json
import os

import numpy as np
import pandas as pd


def is_valid_number_array(arr) -> bool:
    """Return True if *arr* is a non-None array of finite numeric values."""
    if arr is None:
        return False
    try:
        arr = np.array(arr, dtype=float)
        return bool(np.all(np.isfinite(arr)))
    except (ValueError, TypeError):
        return False


def main() -> None:
    from emdb.client import EMDB
    from tqdm import tqdm

    client = EMDB()

    results = client.csv_search(
        'half_map_filename:[* TO *] AND current_status:"REL"'
        " AND release_date:[2002-01-01T00:00:00Z TO 2025-06-30T23:59:59Z]"
        " AND database:EMDB"
    )
    all_ids = results["emdb_id"].astype(str).tolist()

    data_file = "fsc_curves_partial.json"
    data_list: list[dict] = []

    if os.path.exists(data_file):
        with open(data_file) as f:
            loaded_data = json.load(f)
        for entry in loaded_data:
            for col in ("fsc_corrected", "fsc_masked", "fsc_phaserandom", "fsc_unmasked"):
                if col in entry:
                    curve = entry[col]
                    try:
                        entry[col] = np.array(curve, dtype=float) if curve is not None else np.array([])
                    except Exception:
                        entry[col] = np.array([])
            data_list.append(entry)

    processed_ids = {e["id"] for e in data_list}
    unprocessed_ids = [i for i in all_ids if i not in processed_ids]
    print(f"Resuming: {len(data_list)} entries already processed, {len(unprocessed_ids)} remaining.")

    BATCH_SIZE = 50
    batch: list[dict] = []

    for emdb_id in tqdm(unprocessed_ids, desc="Processing EMDB entries", unit="entry"):
        try:
            entry = client.get_entry(emdb_id)

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

            all_data = [resolution, fsc_corrected, fsc_masked, fsc_phaserandom, fsc_unmasked]
            if not all(is_valid_number_array(d) for d in all_data):
                continue

            batch.append({
                "id": entry.id,
                "method": entry.method,
                "resolution": resolution,
                "fsc_corrected": fsc_corrected,
                "fsc_phaserandom": fsc_phaserandom,
                "fsc_masked": fsc_masked,
                "fsc_unmasked": fsc_unmasked,
            })

            if len(batch) >= BATCH_SIZE:
                data_list.extend(batch)
                with open(data_file, "w") as f:
                    json.dump(data_list, f)
                batch.clear()
                print(f"Checkpoint saved ({len(data_list)} entries total)")

        except Exception as e:
            print(f"Failed to process {emdb_id}: {e}")

    if batch:
        data_list.extend(batch)
        with open(data_file, "w") as f:
            json.dump(data_list, f)

    columns = ["id", "method", "resolution", "fsc_corrected", "fsc_phaserandom", "fsc_masked", "fsc_unmasked"]
    fsc_df = pd.DataFrame(data_list, columns=columns)  # noqa: F841 – available for interactive use

    with open("data/fsc_curves_all.json", "w") as f:
        json.dump(data_list, f)

    print(f"Done. {len(data_list)} entries exported.")


if __name__ == "__main__":
    main()