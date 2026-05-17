from pathlib import Path

import pandas as pd

_BASE = Path(__file__).resolve().parent

# List your CSV files here
csv_files = ['Processed_Education2023.csv', 'Processed_Poverty2023.csv', 'Processed_Unemployment2023.csv', 'Processed_PopulationEstimates.csv']

# Initialize the merged DataFrame with the first CSV file
merged_df = pd.read_csv(_BASE / csv_files[0])

# Loop through the remaining CSV files and merge them on 'fips_code'
for file in csv_files[1:]:
    df = pd.read_csv(_BASE / file)
    merged_df = pd.merge(merged_df, df, on='fips_code', how='outer')

out_path = _BASE / 'DemographicsData.csv'
merged_df.to_csv(out_path, index=False)
print(f"Merged CSV file saved as '{out_path}'")
