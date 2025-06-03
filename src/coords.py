import geopandas as gpd
import pandas as pd
import numpy as np

data_path = './data/DeadTrees_2023_Anis_ShapeStudy.gpkg'

# Load the GeoPackage file
data = gpd.read_file(data_path)

# Check Coordinate Reference System (CRS) and reproject if necessary
if data.crs.is_geographic:
    # Example: Reproject to UTM Zone 10N (adjust based on data location)
    data = data.to_crs('EPSG:32610')

# Extract coordinates for clustering
coords = pd.DataFrame({'x': data.geometry.x, 'y': data.geometry.y})

# Save coordinates to CSV file
coords.to_csv('./output/tree_coordinates.csv', index=False)

'''
Usage:

python ./src/coords.py

scp output/tree_coordinates.csv rahmanan@lumi.csc.fi:/scratch/project_462000684/rahmanan/tree_clusters/output/

'''