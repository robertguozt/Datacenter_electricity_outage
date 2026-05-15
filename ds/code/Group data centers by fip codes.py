from pathlib import Path

import pandas as pd

_DS = Path(__file__).resolve().parent

# Load the merged data
merge_ba_carbon_with_fips = pd.read_csv(_DS / 'Merge_BA_Carbon_with_FIPS.csv')

# Ensure fips_code is an integer (handle any NaN or float values)
merge_ba_carbon_with_fips['fips_code'] = merge_ba_carbon_with_fips['fips_code'].astype('Int64')

# Group by FIPS code and aggregate into lists
grouped_by_fips = merge_ba_carbon_with_fips.groupby('fips_code').agg(
    data_centers=('Data_center', list),  # Collect data center names into a list
    total_power_usage_kwh=('power_usage_2023_kwh', list),  # Collect power usage into a list
    total_carbon_emission_2023_tons=('carbon_emission_2023_tons', list)  # Collect carbon emission into a list
).reset_index()

# Convert fips_code to integer after grouping (handling potential NaN)
grouped_by_fips['fips_code'] = grouped_by_fips['fips_code'].astype('Int64')

# Save the grouped data to a new CSV file
grouped_by_fips.to_csv(_DS / 'Grouped_By_FIPS_List.csv', index=False)

# Print the first few rows of the grouped data
print(grouped_by_fips.head(10).to_string())
print(grouped_by_fips.info())
