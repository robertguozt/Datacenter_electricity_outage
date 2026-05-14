import ast
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent

# Load the datasets (paths anchored to repo root)
outage_df = pd.read_csv(ROOT / 'Outage' / 'Aggregated_Outage_Events.csv')
grouped_df = pd.read_csv(ROOT / 'DS' / 'Grouped_By_FIPS_List.csv')

# Convert 'durations_hrs' and 'sums' from string to list of floats
outage_df['durations_hrs'] = outage_df['durations_hrs'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')] if pd.notnull(x) else [])
outage_df['sums'] = outage_df['sums'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')] if pd.notnull(x) else [])

# Convert 'total_power_usage_kwh' and 'total_carbon_emission_2023_tons' from string to list of floats
grouped_df['total_power_usage_kwh'] = grouped_df['total_power_usage_kwh'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')] if pd.notnull(x) else [])
grouped_df['total_carbon_emission_2023_tons'] = grouped_df['total_carbon_emission_2023_tons'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')] if pd.notnull(x) else [])

def _len_data_centers(cell):
    if pd.isna(cell) or (isinstance(cell, str) and not cell.strip()):
        return 0
    try:
        return len(ast.literal_eval(cell))
    except (ValueError, SyntaxError, TypeError):
        return 0


grouped_df['number_of_data_centers'] = grouped_df['data_centers'].apply(_len_data_centers)

# ----------------- First Correlation Matrix -----------------
# Merge the two datasets on 'fips_code' (inner join: only counties with data centers)
merged_df_inner = pd.merge(outage_df, grouped_df, on='fips_code', how='inner')

# Flatten the lists by taking the mean
merged_df_inner['avg_durations_hrs'] = merged_df_inner['durations_hrs'].apply(lambda x: sum(x)/len(x) if len(x) > 0 else 0)
merged_df_inner['sum_duration_hrs'] = merged_df_inner['durations_hrs'].apply(lambda x: sum(x) if len(x) > 0 else 0)
merged_df_inner['sums_sum'] = merged_df_inner['sums'].apply(lambda x: sum(x) if len(x) > 0 else 0)
merged_df_inner['avg_power_usage'] = merged_df_inner['total_power_usage_kwh'].apply(lambda x: sum(x)/len(x) if len(x) > 0 else 0)
merged_df_inner['avg_carbon_emission'] = merged_df_inner['total_carbon_emission_2023_tons'].apply(lambda x: sum(x)/len(x) if len(x) > 0 else 0)

# Select relevant columns
set1_inner = merged_df_inner[['avg_durations_hrs', 'sum_duration_hrs', 'sums_sum', 'counts_of_outage', 'average_sums']]
set2_inner = merged_df_inner[['number_of_data_centers', 'avg_power_usage', 'avg_carbon_emission']]
combined_inner = pd.concat([set1_inner, set2_inner], axis=1)

# Correlation matrix for inner join
correlation_matrix_inner = combined_inner.corr()

# ----------------- Second Correlation Matrix -----------------
merged_df_full = pd.merge(outage_df, grouped_df, on='fips_code', how='left')
merged_df_full['number_of_data_centers'] = merged_df_full['number_of_data_centers'].fillna(0)
merged_df_full['total_power_usage_kwh'] = merged_df_full['total_power_usage_kwh'].apply(lambda x: [0] if isinstance(x, float) and pd.isna(x) else x)
merged_df_full['total_carbon_emission_2023_tons'] = merged_df_full['total_carbon_emission_2023_tons'].apply(lambda x: [0] if isinstance(x, float) and pd.isna(x) else x)

merged_df_full['avg_durations_hrs'] = merged_df_full['durations_hrs'].apply(lambda x: sum(x)/len(x) if len(x) > 0 else 0)
merged_df_full['sum_duration_hrs'] = merged_df_full['durations_hrs'].apply(lambda x: sum(x) if len(x) > 0 else 0)
merged_df_full['sums_sum'] = merged_df_full['sums'].apply(lambda x: sum(x) if len(x) > 0 else 0)
merged_df_full['avg_power_usage'] = merged_df_full['total_power_usage_kwh'].apply(lambda x: sum(x)/len(x) if len(x) > 0 else 0)
merged_df_full['avg_carbon_emission'] = merged_df_full['total_carbon_emission_2023_tons'].apply(lambda x: sum(x)/len(x) if len(x) > 0 else 0)

set1_full = merged_df_full[['avg_durations_hrs', 'sum_duration_hrs', 'sums_sum', 'counts_of_outage', 'average_sums']]
set2_full = merged_df_full[['number_of_data_centers', 'avg_power_usage', 'avg_carbon_emission']]
combined_full = pd.concat([set1_full, set2_full], axis=1)

correlation_matrix_full = combined_full.corr()

# ----------------- Visualization -----------------
def plot_correlation(corr_matrix, title, sample_size, set1_columns, set2_columns):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
    plt.title(f'{title} (Sample Size: {sample_size})', fontsize=14)
    ax = plt.gca()
    set1_color, set2_color = 'blue', 'green'
    for label in ax.get_xticklabels():
        label.set_color(set1_color if label.get_text() in set1_columns else set2_color)
    for label in ax.get_yticklabels():
        label.set_color(set1_color if label.get_text() in set1_columns else set2_color)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

set1_columns = ['avg_durations_hrs', 'sum_duration_hrs', 'sums_sum', 'counts_of_outage', 'average_sums']
set2_columns = ['number_of_data_centers', 'avg_power_usage', 'avg_carbon_emission']

plot_correlation(correlation_matrix_inner, "Pearson Correlation (Only counties with data center)", combined_inner.shape[0], set1_columns, set2_columns)
plot_correlation(correlation_matrix_full, "Pearson Correlation (All counties)", combined_full.shape[0], set1_columns, set2_columns)
