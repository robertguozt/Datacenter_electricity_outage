from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

_DS = Path(__file__).resolve().parent

# Load the data with specified encoding
merge_ba_carbon = pd.read_csv(_DS / 'Merge_BA_Carbon.csv', encoding='ISO-8859-1')

# Ensure latitude and longitude are converted to floats
merge_ba_carbon['Longitude_of_data_center'] = pd.to_numeric(merge_ba_carbon['Longitude_of_data_center'], errors='coerce')
merge_ba_carbon['Latitude_of_data_center'] = pd.to_numeric(merge_ba_carbon['Latitude_of_data_center'], errors='coerce')

# Drop rows with invalid coordinates
merge_ba_carbon = merge_ba_carbon.dropna(subset=['Longitude_of_data_center', 'Latitude_of_data_center'])

# Convert the data center coordinates into GeoDataFrame
merge_ba_carbon['geometry'] = merge_ba_carbon.apply(
    lambda row: Point(row['Longitude_of_data_center'], row['Latitude_of_data_center']), axis=1
)
data_centers_gdf = gpd.GeoDataFrame(merge_ba_carbon, geometry='geometry', crs='EPSG:4326')

# Load the shapefile of US counties (replace with the actual path to your shapefile)
counties_gdf = gpd.read_file(_DS / 'tl_2024_us_county.shp')

# Print columns to find the FIPS code column name
print(counties_gdf.columns)

# Reproject counties to the same CRS as data centers
counties_gdf = counties_gdf.to_crs(data_centers_gdf.crs)

# Perform a spatial join to find the county for each data center
data_centers_with_fips = gpd.sjoin(data_centers_gdf, counties_gdf, how='left', predicate='within')

# Replace 'GEOID' with the actual column name for FIPS codes
merge_ba_carbon['fips_code'] = data_centers_with_fips['GEOID']  # Update based on the column name in your shapefile

# Save the resulting DataFrame
merge_ba_carbon.to_csv(_DS / 'Merge_BA_Carbon_with_FIPS.csv', index=False)

print(merge_ba_carbon.head())
print(merge_ba_carbon.info())