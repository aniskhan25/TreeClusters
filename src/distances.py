import os
import hashlib
import logging
import argparse
import rasterio

import numpy as np
import pandas as pd
import geopandas as gpd

from tqdm import tqdm  # For progress bar
from scipy.ndimage import binary_closing, binary_opening, binary_erosion, distance_transform_edt
from skimage.measure import label
from skimage.morphology import remove_small_holes
from rasterio.warp import reproject, Resampling  # Explicitly import warp functions
from multiprocessing import Pool, cpu_count


logger = logging.getLogger(__name__)


def chamfer_distance_transform(binary_image, weights=(3, 4)):
    h_weight, d_weight = weights
    dist = np.where(binary_image == 0, 0, np.inf)
    rows, cols = dist.shape
    for i in range(rows):
        for j in range(cols):
            if dist[i, j] == np.inf:
                neighbors = []
                if i > 0:
                    neighbors.append(dist[i - 1, j] + h_weight)
                    if j > 0:
                        neighbors.append(dist[i - 1, j - 1] + d_weight)
                    if j < cols - 1:
                        neighbors.append(dist[i - 1, j + 1] + d_weight)
                if j > 0:
                    neighbors.append(dist[i, j - 1] + h_weight)
                if neighbors:
                    dist[i, j] = min(neighbors)
    for i in range(rows - 1, -1, -1):
        for j in range(cols - 1, -1, -1):
            if dist[i, j] != 0:
                neighbors = [dist[i, j]]
                if i < rows - 1:
                    neighbors.append(dist[i + 1, j] + h_weight)
                    if j > 0:
                        neighbors.append(dist[i + 1, j - 1] + d_weight)
                    if j < cols - 1:
                        neighbors.append(dist[i + 1, j + 1] + d_weight)
                if j < cols - 1:
                    neighbors.append(dist[i, j + 1] + h_weight)
                dist[i, j] = min(neighbors)
    return dist


def distance_to_forest_edge(
    longitude,
    latitude,
    vmi_raster_path,
    dem_raster_path,
    window_size=400,
    threshold=30,
    min_patch_pixels=20,
    max_hole_pixels=20,
    weight_range=(1.05, 1.10),
    south_facing_range=(135, 225),
):
    with rasterio.open(vmi_raster_path) as vmi_src, rasterio.open(dem_raster_path) as dem_src:
        row, col = vmi_src.index(longitude, latitude)
        window = rasterio.windows.Window(col - window_size // 2, row - window_size // 2, window_size, window_size)
        canopy_cover = vmi_src.read(1, window=window)
        vmi_transform = vmi_src.window_transform(window)
        vmi_pixel_size = vmi_src.res[0]
        dem_window = dem_src.window(*vmi_src.window_bounds(window))
        dem = dem_src.read(1, window=dem_window)
        dem_transform = dem_src.window_transform(dem_window)
        dem_resampled = np.zeros_like(canopy_cover, dtype=np.float32)
        reproject(
            source=dem,
            destination=dem_resampled,
            src_transform=dem_transform,
            src_crs=dem_src.crs,
            dst_transform=vmi_transform,
            dst_crs=vmi_src.crs,
            resampling=Resampling.bilinear,
        )
        dy, dx = np.gradient(dem_resampled, vmi_pixel_size)
        aspect_rad = np.arctan2(-dy, dx)
        aspect_deg = np.degrees(aspect_rad) % 360
        initial_forest_mask = ((canopy_cover != 32767) & (canopy_cover >= threshold)).astype(np.uint8)
        forest_mask = binary_closing(initial_forest_mask, structure=np.ones((3, 3)))
        forest_mask = binary_opening(forest_mask, structure=np.ones((3, 3)))
        labeled_forest, num_features = label(forest_mask, return_num=True)
        cleaned_forest_mask = np.zeros_like(forest_mask)
        for i in range(1, num_features + 1):
            component = labeled_forest == i
            if np.sum(component) >= min_patch_pixels:
                cleaned_forest_mask[component] = 1
        filled_forest_mask = remove_small_holes(cleaned_forest_mask.astype(bool), area_threshold=max_hole_pixels)
        forest_mask = filled_forest_mask.astype(np.uint8)
        eroded_forest = binary_erosion(forest_mask, structure=np.ones((3, 3)))
        forest_edge = forest_mask ^ eroded_forest
        dist_all = chamfer_distance_transform(1 - forest_edge)
        south_facing_mask = (aspect_deg >= south_facing_range[0]) & (aspect_deg <= south_facing_range[1])
        south_facing_edges = forest_edge & south_facing_mask
        dist_south = chamfer_distance_transform(1 - south_facing_edges)
        aspect_center = (south_facing_range[0] + south_facing_range[1]) / 2
        aspect_deviation = np.abs(aspect_deg - aspect_center) / ((south_facing_range[1] - south_facing_range[0]) / 2)
        weight_factor = np.where(
            south_facing_mask,
            weight_range[0] + (weight_range[1] - weight_range[0]) * (1 - np.minimum(aspect_deviation, 1)),
            1.0,
        )
        adjusted_dist = np.where(dist_south == dist_all, dist_all * weight_factor, dist_all)
        adjusted_dist *= vmi_pixel_size
        # adjusted_dist = np.minimum(adjusted_dist, 300)  # Cap at patch size
        row_in_window = row - int(window.row_off)
        col_in_window = col - int(window.col_off)
        return adjusted_dist[row_in_window, col_in_window]


def distance_to_nearest_wetland(dtw_path, longitude, latitude, wetland_threshold=1):
    try:
        with rasterio.open(dtw_path) as src:
            dtw = src.read(1, masked=True)

            pixel_size = src.res[0]
            rows, cols = dtw.shape

            row, col = src.index(longitude, latitude)

            row = max(0, min(row, rows - 1))
            col = max(0, min(col, cols - 1))

            # DTW index threshold (in meters) below which a pixel is considered wet.
            # Default is 1 (i.e., pixels with DTW <1m are "wet").
            wetland_mask = (dtw < wetland_threshold).astype(np.uint8)

            distance_to_wetland = distance_transform_edt(1 - wetland_mask, sampling=[pixel_size, pixel_size])

            distance_to_wetland = np.maximum(distance_to_wetland, 1)

            return distance_to_wetland[row, col]
    except Exception as e:
        logger.error(f"Error computing distance: {e}")
        return None


def distance_to_rocky_outcrop(dem_path, target_longitude, target_latitude, rock_threshold=30):
    try:
        with rasterio.open(dem_path) as src:
            dem = src.read(1, masked=True)
            pixel_size = src.res[0]
            rows, cols = dem.shape  # Get actual patch dimensions
            row, col = src.index(target_longitude, target_latitude)
            row = max(0, min(row, rows - 1))
            col = max(0, min(col, cols - 1))
            dy, dx = np.gradient(dem.astype("float"), pixel_size)
            slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
            slope_deg = np.degrees(slope_rad)
            rocky_mask = slope_deg > rock_threshold
            rocky_mask = binary_closing(rocky_mask, structure=np.ones((3, 3)))
            distance_from_rock = distance_transform_edt(1 - rocky_mask, sampling=[pixel_size, pixel_size])
            return distance_from_rock[row, col]
    except Exception as e:
        logger.error(f"Error in distance_to_rocky_outcrop: {e}")
        return None


def compute_all_distances(args):
    row, output_dir = args
    results = {}

    # If filename is missing, log and proceed with individual checks
    tif_filename = row['Filename']
    if pd.isna(tif_filename):
        logger.debug(f"No TIFF filename recorded for patch {row['patch_id']}. Proceeding with available patches.")
    else:
        # Define patch paths
        candidate_vmi = os.path.join(output_dir, "vmi", tif_filename)
        candidate_dem = os.path.join(output_dir, "dem", tif_filename)
        candidate_wl = os.path.join(output_dir, "dtw", tif_filename)

        # Forest edge distance (requires both VMI and DEM)
        if not (os.path.exists(candidate_vmi) and os.path.exists(candidate_dem)):
            logger.warning(f"TIFF files missing for forest edge at {candidate_vmi} or {candidate_dem}.")
            results['distance_to_forest_edge'] = None
        else:
            try:
                results['distance_to_forest_edge'] = distance_to_forest_edge(
                    row['x'], row['y'], candidate_vmi, candidate_dem, window_size=20
                )
            except Exception as e:
                logger.error(f"Error computing forest edge distance {tif_filename}: {e}")
                results['distance_to_forest_edge'] = None

        # Rocky outcrop distance
        if not os.path.exists(candidate_dem):
            logger.warning(f"TIFF file for rocky outcrop not found at {candidate_dem}.")
            results['distance_to_rocky_outcrop'] = None
        else:
            try:
                results['distance_to_rocky_outcrop'] = distance_to_rocky_outcrop(candidate_dem, row['x'], row['y'])
            except Exception as e:
                logger.error(f"Error computing rocky outcrop distance: {e}")
                results['distance_to_rocky_outcrop'] = None

        # Wetland distance
        if not os.path.exists(candidate_wl):
            logger.warning(f"TIFF file for wetland not found at {candidate_wl}.")
            results['distance_to_nearest_wetland'] = None
        else:
            try:
                results['distance_to_nearest_wetland'] = distance_to_nearest_wetland(candidate_wl, row['x'], row['y'])
            except Exception as e:
                logger.error(f"Error computing wetland distance: {e}")
                results['distance_to_nearest_wetland'] = None

    return results


def add_distance_column_to_cluster_df(cluster_df, mapping_csv_path, output_dir, collection_names, num_workers=None):
    mappings = pd.read_csv(mapping_csv_path)
    if 'TreeID' in mappings.columns:
        mappings.rename(columns={'TreeID': 'patch_id'}, inplace=True)
    merged_df = pd.merge(cluster_df, mappings[['patch_id', 'Filename']], on='patch_id', how='left')

    if num_workers is None:
        num_workers = cpu_count()
    logger.info(f"Using {num_workers} CPU workers for parallel processing.")

    rows = [row for _, row in merged_df.iterrows()]
    args = [(row, output_dir) for row in rows]

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(compute_all_distances, args), total=len(merged_df), desc="Computing distances"))

    distances_df = pd.DataFrame(results, index=merged_df.index)
    merged_df = pd.concat([merged_df, distances_df], axis=1)
    return merged_df


def load_cluster_data(data_path):
    cluster_df = pd.read_csv(data_path)

    def generate_patch_id(row):
        unique_str = f"{row['x']}_{row['y']}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:8]

    cluster_df["patch_id"] = cluster_df.apply(generate_patch_id, axis=1)
    return cluster_df


def main():
    parser = argparse.ArgumentParser(description="Process tree cluster data and extract patches.")
    parser.add_argument("--data-path", required=True, help="Path to the input data file (e.g., .gpkg or .csv).")
    parser.add_argument("--output-dir", required=True, help="Directory to save the output patches and mapping file.")
    args = parser.parse_args()

    data_path = args.data_path
    output_dir = args.output_dir

    collection_names = ["dtw", "dem", "vmi"]
    mapping_csv_path = os.path.join(output_dir, "mapping.csv")

    cluster_df = load_cluster_data(data_path)
    cluster_df_with_distance = add_distance_column_to_cluster_df(
        cluster_df, mapping_csv_path, output_dir, collection_names
    )

    output_csv = os.path.join(output_dir, "clusters_with_distance.csv")
    cluster_df_with_distance.to_csv(output_csv, index=False)
    logger.info(f"Saved updated cluster data with distances to {output_csv}")


if __name__ == "__main__":
    main()

'''

Usage:

python ./src/distances.py --data-path ./data/clusters.csv --output-dir ./output

sbatch ~/TreeClusters/scripts/run_distances.sh lumi

'''
