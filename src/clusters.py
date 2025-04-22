
import numpy as np
import pandas as pd

import geopandas as gpd
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint
import matplotlib.pyplot as plt


from scipy.spatial import cKDTree, ConvexHull
from sklearn.cluster import DBSCAN


def find_clusters(df, eps=20, min_samples=3):
    coords = df[["x", "y"]].values
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)

    df["cluster"] = db.labels_

    df["event_type"] = df["cluster"].apply(lambda x: "Isolated" if x == -1 else "Clustered")

    clustered_count = df[df["event_type"] == "Clustered"].shape[0]
    isolated_count = df[df["event_type"] == "Isolated"].shape[0]

    print(f"Number of clustered events: {clustered_count}")
    print(f"Number of isolated events: {isolated_count}")

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
    # Load the GeoPackage file
    data = gpd.read_file('./data/DeadTrees_2023_Anis_ShapeStudy.gpkg')

    # Check Coordinate Reference System (CRS) and reproject if necessary
    if data.crs.is_geographic:
        # Example: Reproject to UTM Zone 10N (adjust based on data location)
        data = data.to_crs('EPSG:32610')

    # Extract coordinates for clustering
    coords = pd.DataFrame({'x': data.geometry.x, 'y': data.geometry.y})

    df_clusters = find_clusters(coords, eps=20, min_samples=3)

    df_clusters.to_csv('./data/clusters.csv', index=False)

if __name__ == "__main__":
    main()