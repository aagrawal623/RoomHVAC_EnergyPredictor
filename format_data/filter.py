import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────
# Update this to the path of the CSV file with the combined Smart Building HVAC data
FILE_PATH = ''

# Keywords of columns to keep. Any column containing any of these strings will be kept
COLUMN_KEYWORDS = [
    'Timestamp',
    'hhw valve position',           # reheat proxy
    'supply air flow (%)',
    'supply air flow cfm',          # primary energy driver
    'supply air flow clg max sp',   # normalization denominator (cfm / clg_max = % effort)
    'supply air temp',              # needed for cooling load: Q = cfm × 1.08 × (zone_temp - supply_temp)
    'zone temp',                    # actual zone temperature + all setpoints
    'zone temp eff sp',             # what the zone is actually targeting
    'zone temp clg loop output',    # how hard cooling is working (0-100%)
    'zone temp htg loop output',    # how hard heating is working (0-100%)
    'occ schedule status',          # occupied vs unoccupied context
    'hvac mode status',             # heating / cooling / deadband mode
    'supply air damper position',   # damper tracking & mechanical health
]
# ───────────────────────────────────────────────────────────────────────────────

def filter_csv(file_path: str) -> None:
    """
    Creates a new CSV file that contains only columns that contain the strings in COLUMN_KEYWORDS.
    :param file_path: the path of the CSV file to filter
    """
    df = pd.read_csv(file_path, low_memory=False)

    # Keep column if it contains ANY of the keywords (case-insensitive)
    cols_to_keep = [
        col for col in df.columns
        if any(kw.lower() in col.lower() for kw in COLUMN_KEYWORDS)
    ]

    print(f"Columns kept:   {len(cols_to_keep)}")
    print(f"Columns dropped: {len(df.columns) - len(cols_to_keep)}")
    print("\nKept columns:")
    for col in cols_to_keep:
        print(f"  {col}")

    df_filtered = df[cols_to_keep]
    df_filtered.to_csv(file_path.replace(".csv", "_filtered.csv"), index=False)
    print("\nSaved!")


if __name__ == "__main__":
    filter_csv(FILE_PATH)