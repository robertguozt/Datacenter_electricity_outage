from pathlib import Path

import pandas as pd

_BASE = Path(__file__).resolve().parent
input_file = _BASE / 'PopulationEstimates.csv'
output_file = _BASE / 'Processed_PopulationEstimates.csv'

try:
    # Read the CSV file with encoding handling
    df = pd.read_csv(input_file, encoding='ISO-8859-1', dtype=str)

    # Ensure necessary columns exist
    required_columns = {'FIPStxt', 'Attribute'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"One or more required columns {required_columns} are missing in the CSV file.")

    # Filter rows where 'Attribute' contains '2023'
    df_filtered = df[df['Attribute'].str.contains('2023', na=False)]

    # Reshape the dataframe to have 'FIPStxt' as key and 'Attribute' values as columns
    df_pivoted = df_filtered.pivot(index='FIPStxt', columns='Attribute', values='Value')

    # Reset index to flatten the dataframe
    df_pivoted.reset_index(inplace=True)

    # Rename the key column for clarity
    df_pivoted.rename(columns={'FIPStxt': 'fips_code'}, inplace=True)

    # Save the processed data to a new CSV file
    df_pivoted.to_csv(output_file, index=False, encoding='utf-8')

    print(f"Processed data saved to {output_file}")

except UnicodeDecodeError:
    print("Error: Unable to decode the file. Try using a different encoding such as 'latin1' or 'ISO-8859-1'.")
except FileNotFoundError:
    print("Error: The specified file was not found. Please check the file path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
