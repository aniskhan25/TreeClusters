import os
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree, ConvexHull
from shapely.geometry import MultiPoint
from sklearn.cluster import DBSCAN


def find_clusters(df, eps=20, min_samples=3):
    coords = df[["x", "y"]].values
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)

    df["cluster"] = db.labels_

    df["event_type"] = df["cluster"].apply(lambda x: "Isolated" if x == -1 else "Clustered")

    clustered_count = df[df["event_type"] == "Clustered"].shape[0]
    isolated_count = df[df["event_type"] == "Isolated"].shape[0]

    return df


def compute_nearest_neighbor_distance(df: pd.DataFrame) -> pd.DataFrame:
    coords = df[['x', 'y']].values

    tree = cKDTree(coords)

    distances, _ = tree.query(coords, k=2)  # k=2 because the first nearest neighbor is the point itself
    
    df['nearest_neighbor_distance'] = distances[:, 1]

    return df


def compute_density_within_radius(df: pd.DataFrame, radius: float) -> pd.DataFrame:
    coords = df[['x', 'y']].values
    
    tree = cKDTree(coords)
    
    neighbors_within_radius = tree.query_ball_point(coords, r=radius)
    
    area = np.pi * radius**2
    densities = [len(neighbors) / area for neighbors in neighbors_within_radius]

    df[f'density_within_{radius}m'] = densities

    return df


def compute_dbscan_clustering(df: pd.DataFrame, eps: float = 10, min_samples: int = 5) -> pd.DataFrame:
    coords = df[['x', 'y']].values

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(coords)

    df['cluster_id'] = cluster_labels

    cluster_sizes = df.groupby('cluster_id').size().to_dict()
    df['cluster_size'] = df['cluster_id'].map(cluster_sizes)

    def calculate_cluster_area(cluster_coords):
        if len(cluster_coords) > 2:
            hull = ConvexHull(cluster_coords)
            return hull.area
        return 0  # Not enough points to form a convex hull

    cluster_areas = df.groupby('cluster_id').apply(lambda x: calculate_cluster_area(x[['x', 'y']].values)).to_dict()
    df['cluster_area'] = df['cluster_id'].map(cluster_areas)

    df['average_cluster_area'] = df['cluster_area'] / df['cluster_size']

    return df

def main():
    parser = argparse.ArgumentParser(description="Process tree cluster data and extract patches.")
    parser.add_argument("--data-path", required=True, help="Path to the input data file (e.g., .gpkg or .csv).")
    parser.add_argument("--output-dir", required=True, help="Directory to save the output patches and mapping file.")
    args = parser.parse_args()

    data_path = args.data_path
    output_dir = args.output_dir

    # Load the GeoPackage file
    data = gpd.read_file(data_path)

    # Check Coordinate Reference System (CRS) and reproject if necessary
    if data.crs.is_geographic:
        # Example: Reproject to UTM Zone 10N (adjust based on data location)
        data = data.to_crs('EPSG:32610')

    # Extract coordinates for clustering
    coords = pd.DataFrame({'x': data.geometry.x, 'y': data.geometry.y})

    df_clusters = find_clusters(coords, eps=20, min_samples=3)

    # Save the labeled points with cluster information
    df_clusters.to_csv(os.path.join(output_dir, 'cluster_data.csv'), index=False)

    # Compute cluster centroids
    clustered = df_clusters[df_clusters['cluster'] != -1]
    cluster_centroids = clustered.groupby('cluster')[['x', 'y']].mean().reset_index()
    cluster_centroids['event_type'] = 'Clustered'

    print(cluster_centroids.head())

    isolated_events = df_clusters[df_clusters['cluster'] == -1][['x', 'y']].copy()
    isolated_events['event_type'] = 'Isolated'

    # Ensure 'cluster' column is present in both DataFrames
    cluster_centroids = cluster_centroids.rename(columns={'cluster': 'cluster'})
    isolated_events['cluster'] = -1

    final_events = pd.concat([cluster_centroids[['cluster', 'x', 'y', 'event_type']],
                              isolated_events[['cluster', 'x', 'y', 'event_type']]], ignore_index=True)
    final_events.to_csv(os.path.join(output_dir, 'clusters.csv'), index=False)

    print(f"Number of clustered trees: {len(clustered)}")
    print(f"Number of isolated trees: {len(isolated_events)}")

    print(f"Number of clustered events: {len(cluster_centroids)}")
    print(f"Number of isolated events: {len(isolated_events)}")


if __name__ == "__main__":
    main()


'''

Usage:

python ./src/clusters.py --data-path ./data/DeadTrees_2023_Anis_ShapeStudy.gpkg --output-dir ./output

scp data/clusters.csv rahmanan@lumi.csc.fi:/scratch/project_462000684/rahmanan/tree_clusters/data/

'''
