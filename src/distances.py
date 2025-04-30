import os
import hashlib
import logging
import argparse
import rasterio
import importlib

import numpy as np
import pandas as pd
import geopandas as gpd

from scipy.ndimage import binary_closing, binary_opening, binary_erosion, distance_transform_edt
from skimage.measure import label
from skimage.morphology import remove_small_holes
try:
    from rasterio.warp import reproject
except ImportError:
    from rasterio._warp import reproject
from rasterio.enums import Resampling

from tqdm import tqdm
from multiprocessing import Pool, cpu_count


logger = logging.getLogger(__name__)


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
    kernel_size=3,
):
    with rasterio.open(vmi_raster_path) as vmi_src, rasterio.open(dem_raster_path) as dem_src:
        # Get row, col indices for the point
        row, col = vmi_src.index(longitude, latitude)

        # Ensure the point is within the raster bounds
        if row < 0 or row >= vmi_src.height or col < 0 or col >= vmi_src.width:
            return np.nan

        # Calculate window boundaries
        half_window = window_size // 2
        window_col_start = max(0, col - half_window)
        window_row_start = max(0, row - half_window)
        window_col_end = min(vmi_src.width, col + half_window)
        window_row_end = min(vmi_src.height, row + half_window)

        # Ensure window has valid dimensions
        if window_col_end <= window_col_start or window_row_end <= window_row_start:
            return np.nan

        # Adjust window size if near boundaries
        window = rasterio.windows.Window(
            window_col_start, window_row_start, window_col_end - window_col_start, window_row_end - window_row_start
        )

        canopy_cover = vmi_src.read(1, window=window)
        vmi_transform = vmi_src.window_transform(window)
        vmi_pixel_size = vmi_src.res[0]

        # Create initial forest mask at native 16 m resolution
        initial_forest_mask = ((canopy_cover != 32767) & (canopy_cover >= threshold)).astype(np.uint8)

        # Read DEM window and transform
        dem_window = dem_src.window(*vmi_src.window_bounds(window))
        dem = dem_src.read(1, window=dem_window)
        dem_transform = dem_src.window_transform(dem_window)
        dem_res = dem_src.res[0]

        # Compute aspect from the DEM at 2 m resolution
        dy, dx = np.gradient(dem, dem_res)
        aspect_rad = np.arctan2(-dy, dx)
        aspect_deg = np.degrees(aspect_rad) % 360
        forest_mask = binary_closing(initial_forest_mask, structure=np.ones((kernel_size, kernel_size)))
        forest_mask = binary_opening(forest_mask, structure=np.ones((kernel_size, kernel_size)))

        labeled_forest, num_features = label(forest_mask, return_num=True)

        cleaned_forest_mask = np.zeros_like(forest_mask)
        for i in range(1, num_features + 1):
            component = labeled_forest == i
            if np.sum(component) >= min_patch_pixels:
                cleaned_forest_mask[component] = 1

        filled_forest_mask = remove_small_holes(cleaned_forest_mask.astype(bool), area_threshold=max_hole_pixels)
        forest_mask = filled_forest_mask.astype(np.uint8)

        eroded_forest = binary_erosion(forest_mask, structure=np.ones((kernel_size, kernel_size)))
        forest_edge = forest_mask ^ eroded_forest
        # If no forest edge present in the patch, return NaN
        if not forest_edge.any():
            return np.nan

        # Hybrid approach: upsample 16 m forest edge to DEM's 2 m grid
        forest_edge_upsamp = np.zeros_like(dem, dtype=np.uint8)
        reproject(
            source=forest_edge.astype(np.uint8),
            destination=forest_edge_upsamp,
            src_transform=vmi_transform,
            src_crs=vmi_src.crs,
            dst_transform=dem_transform,
            dst_crs=dem_src.crs,
            dst_shape=dem.shape,
            resampling=Resampling.bilinear,
        )
        # Binarize after interpolation
        forest_edge_upsamp = forest_edge_upsamp > 0.5

        # Compute Euclidean distance transforms in meters using the DEM's pixel size
        dist_all = distance_transform_edt(1 - forest_edge_upsamp, sampling=[dem_res, dem_res])

        # Mask edges that face south within the specified range
        south_facing_mask = (aspect_deg >= south_facing_range[0]) & (aspect_deg <= south_facing_range[1])
        south_facing_edges_upsamp = forest_edge_upsamp & south_facing_mask
        dist_south = distance_transform_edt(1 - south_facing_edges_upsamp, sampling=[dem_res, dem_res])

        # Compute directional weight factor for south-facing edges
        aspect_center = (south_facing_range[0] + south_facing_range[1]) / 2
        aspect_deviation = np.abs(aspect_deg - aspect_center) / ((south_facing_range[1] - south_facing_range[0]) / 2)

        weight_factor = np.where(
            south_facing_mask,
            weight_range[0] + (weight_range[1] - weight_range[0]) * (1 - np.minimum(aspect_deviation, 1)),
            1.0,
        )

        adjusted_dist = np.where(dist_south == dist_all, dist_all * weight_factor, dist_all)

        # adjusted_dist = np.minimum(adjusted_dist, 300)  # Cap at patch size
        # Compute point index in DEM window
        row_dem, col_dem = dem_src.index(longitude, latitude)
        row_in_window = int(row_dem - dem_window.row_off)
        col_in_window = int(col_dem - dem_window.col_off)
        # Determine maximum searchable distance (half the patch width in meters)
        rows_patch, cols_patch = forest_edge.shape
        max_distance = max(rows_patch, cols_patch) / 2 * dem_res
        dist = adjusted_dist[row_in_window, col_in_window]
        return np.nan if dist >= max_distance else dist


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
            # Fill masked (no-data) pixels with a value above threshold so they are treated as non-wetland
            dtw_filled = dtw.filled(wetland_threshold + 1)
            wetland_mask = (dtw_filled < wetland_threshold).astype(np.uint8)
            # If no wetland pixel in the patch, return NaN
            if not wetland_mask.any():
                return np.nan

            distance_to_wetland = distance_transform_edt(1 - wetland_mask, sampling=[pixel_size, pixel_size])

            # Determine maximum searchable distance (half the patch width in meters)
            rows_patch, cols_patch = dtw.shape
            max_distance = max(rows_patch, cols_patch) / 2 * pixel_size
            dist = distance_to_wetland[row, col]
            return np.nan if dist >= max_distance else dist
    except Exception as e:
        logger.error(f"Error computing distance: {e}")
        return None


def distance_to_rocky_outcrop(dem_path, target_longitude, target_latitude, rock_threshold=30, kernel_size=3):
    try:
        with rasterio.open(dem_path) as src:
            dem = src.read(1, masked=True)

            # Fill masked (no-data) pixels so gradient ignores them
            dem_filled = dem.filled(np.nan)

            pixel_size = src.res[0]
            rows, cols = dem.shape  # Get actual patch dimensions
            row, col = src.index(target_longitude, target_latitude)
            row = max(0, min(row, rows - 1))
            col = max(0, min(col, cols - 1))

            dy, dx = np.gradient(dem_filled.astype("float"), pixel_size)
            slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
            slope_deg = np.degrees(slope_rad)

            rocky_mask = slope_deg > rock_threshold
            rocky_mask = binary_closing(rocky_mask, structure=np.ones((kernel_size, kernel_size)))
            # If no rocky outcrop pixel in the patch, return NaN
            if not rocky_mask.any():
                return np.nan

            distance_from_rock = distance_transform_edt(1 - rocky_mask, sampling=[pixel_size, pixel_size])

            # Determine the maximum searchable distance (half the patch width in meters)
            max_distance = max(rows, cols) / 2 * pixel_size

            dist = distance_from_rock[row, col]
            # Assign NaN if no rocky pixel was found within the patch (i.e., dist == max_distance)
            return np.nan if dist >= max_distance else dist

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
                    row['x'], row['y'], candidate_vmi, candidate_dem
                )
            except Exception as e:
                logger.error(f"Error computing forest edge distance {row['x']} {row['y']} {tif_filename}: {e}")
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
