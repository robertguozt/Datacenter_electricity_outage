from pathlib import Path

import pandas as pd

_BASE = Path(__file__).resolve().parent

# 1. Define a mapping from full state names to postal abbreviations.
state_abbrev = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY'
}

# 2. Read the temperature data
temp_df = pd.read_csv(_BASE / "Temperature Data.csv")
# The temperature data includes columns such as: ID, Name, State, Value, etc.

# 3. Read the county adjacency file (pipe-delimited)
adjacency_df = pd.read_csv(_BASE / "county_adjacency2024.txt", delimiter="|")
# The columns in this file are: "County Name", "County GEOID", "Neighbor Name", "Neighbor GEOID"

# 4. Build a mapping from the county name (as in the adjacency file) to its FIPS code.
# Because counties appear repeatedly (one row per neighbor), we drop duplicates.
county_fips_map = (adjacency_df[['County Name', 'County GEOID']]
                   .drop_duplicates()
                   .set_index('County Name')['County GEOID']
                   .to_dict())

# 5. Create a key in temp_df that matches the county naming in the adjacency file.
# In the temperature file, we have "Name" (e.g., "Autauga County") and "State" (e.g., "Alabama").
# We convert the state name to its postal abbreviation and then form "Autauga County, AL".
temp_df['County Key'] = temp_df['Name'] + ", " + temp_df['State'].map(state_abbrev)

# 6. Map the "County Key" to the FIPS code using our dictionary.
temp_df['FIPS'] = temp_df['County Key'].map(county_fips_map)
temp_df = temp_df.dropna(subset=['FIPS'])
temp_df['FIPS'] = temp_df['FIPS'].astype(int)

# (Optional) If you no longer need the helper column, you can drop it.
temp_df.drop(columns='County Key', inplace=True)

# 7. Save the modified DataFrame to a new CSV file or inspect the results.
print(temp_df.head())
temp_df.to_csv(_BASE / "Temperature Data with FIPS.csv", index=False)
