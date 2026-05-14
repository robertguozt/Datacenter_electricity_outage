import ast
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent

# ============================================================
# 1. Load the Base Datasets (Outage, Grouped, Demographics)
# ============================================================
outage_df = pd.read_csv(ROOT / 'Outage' / 'Aggregated_Outage_Events.csv')
grouped_df = pd.read_csv(ROOT / 'DS' / 'Grouped_By_FIPS_List.csv')
demographics_df = pd.read_csv(ROOT / 'Population' / 'DemographicsData.csv')

print("outage_df:", len(outage_df))
print("grouped_df:", len(grouped_df))
print("demographics_df:", len(demographics_df))

# ============================================================
# 2. Temperature Data Section: Load and Process Temperature Data
# ============================================================
temp_df = pd.read_csv(ROOT / 'Temperature' / 'Temperature Data with FIPS.csv')

# Rename the FIPS column so that it matches the key in the other datasets
temp_df.rename(columns={'FIPS': 'fips_code'}, inplace=True)

# Convert temperature-related columns to numeric (adjust column names as needed)
temp_df['Value'] = pd.to_numeric(temp_df['Value'], errors='coerce')
temp_df['Anomaly (1901-2000 base period)'] = pd.to_numeric(temp_df['Anomaly (1901-2000 base period)'], errors='coerce')
temp_df['Rank'] = pd.to_numeric(temp_df['Rank'], errors='coerce')
temp_df['1901-2000 Mean'] = pd.to_numeric(temp_df['1901-2000 Mean'], errors='coerce')

# ============================================================
# 3. Data Preprocessing Section: Clean and Prepare Datasets
# ============================================================
# Function to convert string representations of lists into lists of floats
def safe_convert_to_list(x):
    try:
        return [float(i) for i in x.strip('[]').split(',')] if isinstance(x, str) else ([x] if isinstance(x, (int, float)) else [])
    except Exception:
        return []

# Process outage_df list columns
outage_df['durations_hrs'] = outage_df['durations_hrs'].apply(safe_convert_to_list)
outage_df['sums'] = outage_df['sums'].apply(safe_convert_to_list)

# Process grouped_df list columns
grouped_df['total_power_usage_kwh'] = grouped_df['total_power_usage_kwh'].apply(safe_convert_to_list)
grouped_df['total_carbon_emission_2023_tons'] = grouped_df['total_carbon_emission_2023_tons'].apply(safe_convert_to_list)

def _len_data_centers(cell):
    if pd.isna(cell) or (isinstance(cell, str) and not cell.strip()):
        return 0
    try:
        return len(ast.literal_eval(cell))
    except (ValueError, SyntaxError, TypeError):
        return 0


grouped_df['number_of_data_centers'] = grouped_df['data_centers'].apply(_len_data_centers)

# Convert demographic fields to numeric where applicable (excluding the fips_code)
demographics_cols = demographics_df.columns.difference(['fips_code'])
demographics_df[demographics_cols] = demographics_df[demographics_cols].apply(pd.to_numeric, errors='coerce')

# ============================================================
# 4. Merge Datasets on 'fips_code'
# ============================================================
# Merge outage and grouped data
merged_df = pd.merge(outage_df, grouped_df, on='fips_code', how='outer')
# Merge demographics data
merged_df = pd.merge(merged_df, demographics_df, on='fips_code', how='outer')
# Merge the temperature data
merged_df = pd.merge(merged_df, temp_df, on='fips_code', how='outer')

# ============================================================
# 5. Flatten List Columns by Taking the Mean (or Sum)
# ============================================================
merged_df['avg_durations_hrs'] = merged_df['durations_hrs'].apply(
    lambda x: sum(x) / len(x) if isinstance(x, list) and len(x) > 0 else 0)
merged_df['sum_duration_hrs'] = merged_df['durations_hrs'].apply(
    lambda x: sum(x) if isinstance(x, list) and len(x) > 0 else 0)
merged_df['sums_sum'] = merged_df['sums'].apply(
    lambda x: sum(x) if isinstance(x, list) and len(x) > 0 else 0)
merged_df['avg_power_usage'] = merged_df['total_power_usage_kwh'].apply(
    lambda x: sum(x) / len(x) if isinstance(x, list) and len(x) > 0 else 0)
merged_df['avg_carbon_emission'] = merged_df['total_carbon_emission_2023_tons'].apply(
    lambda x: sum(x) / len(x) if isinstance(x, list) and len(x) > 0 else 0)

# ============================================================
# 6. Correlation Analysis Section
# ============================================================
# Select only numeric columns for correlation analysis (temperature fields will be included if numeric)
numeric_columns = merged_df.select_dtypes(include=['number', 'float']).columns

# First correlation: Only counties with data centers
merged_df_with_centers = merged_df[merged_df['number_of_data_centers'] > 0]
correlation_matrix_with_centers = merged_df_with_centers[numeric_columns].corr()

# Second correlation: All counties
correlation_matrix_all_counties = merged_df[numeric_columns].corr()

print(merged_df.info())
merged_df = merged_df.drop(['fips_code','start_times', 'end_times','durations_hrs','sums','county','Name','State','ID','state','data_centers','total_power_usage_kwh','total_carbon_emission_2023_tons'], axis=1)
merged_df.to_csv(ROOT / 'Merged for ML.csv', index=False, encoding='utf-8')
# ============================================================
# 7. Visualization Section: Plot Correlation Matrices
# ============================================================
def plot_correlation(corr_matrix, title):
    plt.figure(figsize=(14, 12))  # Increased figure size for clarity
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, fmt=".2f", center=0)
    plt.title(title)
    plt.show()

# Plot correlations for counties with data centers
plot_correlation(
    correlation_matrix_with_centers,
    f"Pearson Correlation (Counties With Data Centers) | Size: {len(merged_df_with_centers)}"
)

# Plot correlations for all counties
plot_correlation(
    correlation_matrix_all_counties,
    f"Pearson Correlation (All Counties) | Size: {len(merged_df)}"
)

# Save the correlation matrices to CSV files
correlation_matrix_with_centers.to_csv(ROOT / 'Correlation_Matrix_Only_Counties_With_Data_Centers.csv')
correlation_matrix_all_counties.to_csv(ROOT / 'Correlation_Matrix_All_Counties.csv')

# Plot the difference in correlations (with vs. without data centers)
correlation_difference = correlation_matrix_with_centers - correlation_matrix_all_counties
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_difference, cmap="coolwarm", annot=True, fmt=".2f", center=0)
plt.title("Difference in Pearson Correlation (With vs. Without Data Centers)")
plt.show()
