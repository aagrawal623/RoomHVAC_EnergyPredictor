import os
import glob
import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────
# Update this to the folder where the weekly Smart Building HVAC CSV files are stored
DATA_DIR = ''

# Update this to the starting name of the specific CSV files to combine (e.g. BrenHall2025, SBSG2025)
FILE_START = 'SBSG2025'
# ────────────────────────────────────────────────────────────────────

def combine_csvs(file_start: str) -> None:
    """
    Creates a CSV file that combines CSV files in the given directory that start with file_start.
    The CSV file will contain only columns found in all CSV files and sorted by Timestamp if present in all files.
    :param file_start: the start of the name of CSV files to combine
    """
    OUTPUT_FILE = os.path.join(DATA_DIR, f"{file_start}_combined.csv")

    # Find all files matching the pattern
    search_pattern = os.path.join(DATA_DIR, f"{file_start}_*.csv")
    file_list = glob.glob(search_pattern)

    # Filter out the combined file just in case it's caught in the glob pattern
    file_list = [f for f in file_list if "combined_output" not in os.path.basename(f)]

    if not file_list:
        print(f"No files matching '{file_start}_*.csv' found in:\n{DATA_DIR}")
        return

    print(f"Found {len(file_list)} files to combine:")
    for f in file_list:
        print(f" - {os.path.basename(f)}")

    # Read all files
    df_list = []
    for file in file_list:
        try:
            df = pd.read_csv(file, low_memory=False)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not df_list:
        print("No data could be read from the files.")
        return

    print("\nConcatenating data (keeping only common columns)...")
    combined_df = pd.concat(df_list, join='inner', ignore_index=True)

    # Sort combined data by Timestamp
    if 'Timestamp' in combined_df.columns:
        print("Sorting chronologically by Timestamp...")
        combined_df['Timestamp'] = combined_df['Timestamp'].str.split(' ').str[0]
        combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], format='mixed', errors='coerce', utc=True)
        combined_df = combined_df.sort_values(by='Timestamp')
    else:
        print("Warning: 'Timestamp' column not found among the common columns.")

    # Save to output file
    print(f"Saving combined dataset to:\n{OUTPUT_FILE}...")
    combined_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Combined file has {len(combined_df):,} rows and {len(combined_df.columns)} columns.")


if __name__ == "__main__":
    combine_csvs(FILE_START)