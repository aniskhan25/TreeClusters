import os
import hashlib
import logging
import argparse
import rasterio

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Sentinel value representing no valid feature found within window
SENTINEL_DISTANCE = 32767  # Sentinel value representing no valid feature found within window

import numpy as np
import pandas as pd

from scipy.ndimage import distance_transform_edt, binary_closing, binary_opening, binary_erosion
from skimage.measure import label as sk_label
from skimage.morphology import remove_small_holes
from rasterio.warp import reproject, Resampling

from tqdm import tqdm
from multiprocessing import Pool, cpu_count



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger.setLevel(logging.DEBUG)


def distance_to_forest_edge(
    longitude,
    latitude,
    vmi_raster_path,
    dem_raster_path,
    window_size_m=400,  # Window size in meters
    threshold=30,
    min_patch_pixels=20,
    max_hole_pixels=20,
    weight_range=(1.05, 1.10),
    south_facing_range=(135, 225),
    kernel_size=3,
    expected_res=0.25  # Expected resolution in meters
):
    """
    Compute the distance from a tree centroid to the nearest forest edge, with adjustments for south-facing edges.
    Uses a windowed approach based on physical distance (meters) rather than pixels.

    Args:
        longitude (float): Longitude of the tree centroid.
        latitude (float): Latitude of the tree centroid.
        vmi_raster_path (str): Path to the VMI canopy cover raster.
        dem_raster_path (str): Path to the DEM raster.
        window_size_m (int): Size of the analysis window in meters (default: 400).
        threshold (int): Canopy cover threshold for forest classification (default: 30).
        min_patch_pixels (int): Minimum pixel count for a forest patch (default: 20).
        max_hole_pixels (int): Maximum pixel count for holes to fill (default: 20).
        weight_range (tuple): Weight range for south-facing edges (default: (1.05, 1.10)).
        south_facing_range (tuple): Aspect range for south-facing edges in degrees (default: (135, 225)).
        kernel_size (int): Size of the structuring element for morphological operations (default: 3).
        expected_res (float): Expected raster resolution in meters (default: 0.25).

    Returns:
        float: Distance to the nearest forest edge in meters, or np.nan if not computable.
    """
    with rasterio.open(vmi_raster_path) as vmi_src, rasterio.open(dem_raster_path) as dem_src:
        # Check resolution consistency
        if abs(vmi_src.res[0] - expected_res) > 0.01:
            logger.warning(f"VMI resolution {vmi_src.res[0]} m differs from expected {expected_res} m.")
        
        # Convert window size from meters to pixels dynamically
        pixel_size = vmi_src.res[0]
        window_size_pixels = int(window_size_m / pixel_size)
        half_window = window_size_pixels // 2

        # Get row, col indices for the point
        row, col = vmi_src.index(longitude, latitude)

        # Ensure the point is within the raster bounds
        if row < 0 or row >= vmi_src.height or col < 0 or col >= vmi_src.width:
            return SENTINEL_DISTANCE

        # Calculate window boundaries
        window_col_start = max(0, col - half_window)
        window_row_start = max(0, row - half_window)
        window_col_end = min(vmi_src.width, col + half_window)
        window_row_end = min(vmi_src.height, row + half_window)

        # Ensure window has valid dimensions
        if window_col_end <= window_col_start or window_row_end <= window_row_start:
            return SENTINEL_DISTANCE

        # Adjust window size if near boundaries
        window = rasterio.windows.Window(
            window_col_start, window_row_start, window_col_end - window_col_start, window_row_end - window_row_start
        )

        # Read canopy cover and DEM within the window
        canopy_cover = vmi_src.read(1, window=window)
        vmi_transform = vmi_src.window_transform(window)
        dem_window = dem_src.window(*vmi_src.window_bounds(window))
        dem = dem_src.read(1, window=dem_window)
        dem_transform = dem_src.window_transform(dem_window)
        dem_resampled = np.zeros_like(canopy_cover, dtype=np.float32)

        # Reproject DEM to match VMI resolution and window
        reproject(
            source=dem,
            destination=dem_resampled,
            src_transform=dem_transform,
            src_crs=dem_src.crs,
            dst_transform=vmi_transform,
            dst_crs=vmi_src.crs,
            resampling=Resampling.bilinear,
        )

        # Compute aspect from DEM
        dy, dx = np.gradient(dem_resampled, pixel_size)
        aspect_rad = np.arctan2(-dy, dx)
        aspect_deg = np.degrees(aspect_rad) % 360

        # Create forest mask
        initial_forest_mask = ((canopy_cover != 32767) & (canopy_cover >= threshold)).astype(np.uint8)
        forest_mask = binary_closing(initial_forest_mask, structure=np.ones((kernel_size, kernel_size)))
        forest_mask = binary_opening(forest_mask, structure=np.ones((kernel_size, kernel_size)))

        # Label and filter small forest patches
        labeled_forest, num_features = sk_label(forest_mask, return_num=True)
        cleaned_forest_mask = np.zeros_like(forest_mask)
        for i in range(1, num_features + 1):
            component = labeled_forest == i
            if np.sum(component) >= min_patch_pixels:
                cleaned_forest_mask[component] = 1

        # Fill small holes in the forest mask
        filled_forest_mask = remove_small_holes(cleaned_forest_mask.astype(bool), area_threshold=max_hole_pixels)
        forest_mask = filled_forest_mask.astype(np.uint8)

        # Detect forest edges
        eroded_forest = binary_erosion(forest_mask, structure=np.ones((kernel_size, kernel_size)))
        forest_edge = forest_mask ^ eroded_forest
        if not forest_edge.any():
            return SENTINEL_DISTANCE

        # Compute distance transform to all edges
        dist_all = distance_transform_edt(1 - forest_edge, sampling=[pixel_size, pixel_size])

        # Adjust distances for south-facing edges
        south_facing_mask = (aspect_deg >= south_facing_range[0]) & (aspect_deg <= south_facing_range[1])
        south_facing_edges = forest_edge & south_facing_mask
        dist_south = distance_transform_edt(1 - south_facing_edges, sampling=[pixel_size, pixel_size])

        aspect_center = (south_facing_range[0] + south_facing_range[1]) / 2
        aspect_deviation = np.abs(aspect_deg - aspect_center) / ((south_facing_range[1] - south_facing_range[0]) / 2)
        weight_factor = np.where(
            south_facing_mask,
            weight_range[0] + (weight_range[1] - weight_range[0]) * (1 - np.minimum(aspect_deviation, 1)),
            1.0,
        )
        adjusted_dist = np.where(dist_south == dist_all, dist_all * weight_factor, dist_all)

        # Get distance at centroid's position in the window
        row_in_window = row - window.row_off
        col_in_window = col - window.col_off
        max_distance = window_size_m / 2
        dist = adjusted_dist[row_in_window, col_in_window]
        return SENTINEL_DISTANCE if dist >= max_distance else dist


import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt
import logging

logger = logging.getLogger(__name__)

def distance_to_nearest_wetland(
    dtw_path, longitude, latitude, wetland_threshold=1, window_size_m=1000, expected_res=0.25
):
    """
    Compute the distance from a tree centroid to the nearest wetland pixel using a windowed approach.

    Args:
        dtw_path (str): Path to the depth-to-water (DTW) raster.
        longitude (float): Longitude of the tree centroid.
        latitude (float): Latitude of the tree centroid.
        wetland_threshold (float): Threshold for wetland classification (default: 1).
        window_size_m (int): Size of the analysis window in meters (default: 1000).
        expected_res (float): Expected raster resolution in meters (default: 0.25).

    Returns:
        float: Distance to the nearest wetland in meters, or np.nan if not computable.
    """
    try:
        with rasterio.open(dtw_path) as src:
            pixel_size = src.res[0]
            if abs(pixel_size - expected_res) > 0.01:
                logger.warning(f"DTW resolution {pixel_size} m differs from expected {expected_res} m.")

            # Convert window size to pixels
            window_size_pixels = int(window_size_m / pixel_size)
            half_window = window_size_pixels // 2

            row, col = src.index(longitude, latitude)
            if row < 0 or row >= src.height or col < 0 or col >= src.width:
                return SENTINEL_DISTANCE

            # Define window bounds
            window_col_start = max(0, col - half_window)
            window_row_start = max(0, row - half_window)
            window_col_end = min(src.width, col + half_window)
            window_row_end = min(src.height, row + half_window)

            if window_col_end <= window_col_start or window_row_end <= window_row_start:
                return SENTINEL_DISTANCE

            window = rasterio.windows.Window(
                window_col_start, window_row_start,
                window_col_end - window_col_start, window_row_end - window_row_start
            )

            dtw = src.read(1, window=window, masked=True)

            # Use raster's nodata value if defined, otherwise mask known invalid values
            if src.nodata is not None:
                dtw = np.ma.masked_equal(dtw, src.nodata)
            else:
                # more robust: also mask 32767, -32768, 65535
                dtw = np.ma.masked_where(np.isin(dtw, [32767, -32768, 65535]), dtw)

            # Safer mask shape check
            if dtw.mask.shape != dtw.shape:
                logger.warning("Mask shape does not match data shape; skipping masking correction to avoid data corruption")

            wetland_mask = np.where((dtw < wetland_threshold) & (~dtw.mask), 1, 0).astype(np.uint8)

            # Diagnostics just before distance_transform_edt
            logger.debug(f"DTW min: {dtw.min()}, max: {dtw.max()}, wetland_threshold: {wetland_threshold}")
            logger.debug(f"Nonzero wetland pixels: {np.count_nonzero(wetland_mask)}")
            if not wetland_mask.any():
                logger.warning("No wetlands detected in the window. Check threshold or raster data.")
                return SENTINEL_DISTANCE

            input_array = 1 - wetland_mask
            logger.debug(f"Distance transform input shape: {input_array.shape}, sampling: {[pixel_size, pixel_size]}, dtype: {input_array.dtype}")
            assert input_array.ndim == 2, "Input array to distance_transform_edt must be 2D"
            assert len([pixel_size, pixel_size]) == input_array.ndim, "Sampling dimensions must match array dimensions"
            try:
                distance_to_wetland = distance_transform_edt(input_array, sampling=[pixel_size, pixel_size])
            except Exception as e:
                logger.exception("Error in distance_transform_edt for wetland mask")
                return SENTINEL_DISTANCE

            max_distance = window_size_m / 2
            row_in_window = row - window.row_off
            col_in_window = col - window.col_off
            dist = distance_to_wetland[row_in_window, col_in_window]
            logger.debug(f"Computed distance to wetland at ({row_in_window}, {col_in_window}): {dist}")
            return SENTINEL_DISTANCE if dist >= max_distance else dist
    except Exception as e:
        logger.error(f"Error computing wetland distance: {e}")
        return None
    
    
def distance_to_rocky_outcrop(
    dem_path, longitude, latitude, rock_threshold=30, kernel_size=3, window_size_m=500, expected_res=0.25
):
    """
    Compute the distance from a tree centroid to the nearest rocky outcrop using a windowed approach.

    Args:
        dem_path (str): Path to the DEM raster.
        longitude (float): Longitude of the tree centroid.
        latitude (float): Latitude of the tree centroid.
        rock_threshold (float): Slope threshold for rocky outcrops in degrees (default: 30).
        kernel_size (int): Size of the structuring element for morphological operations (default: 3).
        window_size_m (int): Size of the analysis window in meters (default: 500).
        expected_res (float): Expected raster resolution in meters (default: 0.25).

    Returns:
        float: Distance to the nearest rocky outcrop in meters, or np.nan if not computable.
    """
    try:
        with rasterio.open(dem_path) as src:
            pixel_size = src.res[0]
            if abs(pixel_size - expected_res) > 0.01:
                logger.warning(f"DEM resolution {pixel_size} m differs from expected {expected_res} m.")

            # Convert window size to pixels
            window_size_pixels = int(window_size_m / pixel_size)
            half_window = window_size_pixels // 2

            row, col = src.index(longitude, latitude)
            if row < 0 or row >= src.height or col < 0 or col >= src.width:
                return SENTINEL_DISTANCE

            # Define window bounds
            window_col_start = max(0, col - half_window)
            window_row_start = max(0, row - half_window)
            window_col_end = min(src.width, col + half_window)
            window_row_end = min(src.height, row + half_window)

            if window_col_end <= window_col_start or window_row_end <= window_row_start:
                return SENTINEL_DISTANCE

            window = rasterio.windows.Window(
                window_col_start, window_row_start,
                window_col_end - window_col_start, window_row_end - window_row_start
            )

            dem = src.read(1, window=window, masked=True)
            row_in_window = row - window.row_off
            col_in_window = col - window.col_off

            # Compute slope
            dy, dx = np.gradient(dem.filled(np.nan).astype("float"), pixel_size)
            slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
            slope_deg = np.degrees(slope_rad)

            # Identify rocky outcrops
            rocky_mask = slope_deg > rock_threshold
            rocky_mask = binary_closing(rocky_mask, structure=np.ones((kernel_size, kernel_size)))
            if not rocky_mask.any():
                return SENTINEL_DISTANCE

            distance_from_rock = distance_transform_edt(1 - rocky_mask, sampling=[pixel_size, pixel_size])
            max_distance = window_size_m / 2
            dist = distance_from_rock[row_in_window, col_in_window]
            return SENTINEL_DISTANCE if dist >= max_distance else dist
    except Exception as e:
        logger.error(f"Error computing rocky outcrop distance: {e}")
        return None


import numpy as np
import rasterio
from scipy.ndimage import label, binary_erosion
import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def compute_additional_features(row, output_dir):
    features = {
        'avg_canopy_cover': np.nan,
        'std_canopy_cover': np.nan,
        'prop_forested_area': np.nan,
        'num_forest_patches': np.nan,
        'edge_density': np.nan,
        'prop_wetland_area': np.nan,
        'avg_dtw': np.nan,
        'std_dtw': np.nan,
        'avg_elevation': np.nan,
        'avg_slope': np.nan,
        'std_slope': np.nan,
        'prop_rocky_outcrops': np.nan
    }
    tif_filename = row['Filename']
    if pd.isna(tif_filename):
        return features

    vmi_path = os.path.join(output_dir, "vmi", tif_filename)
    dtw_path = os.path.join(output_dir, "dtw", tif_filename)
    dem_path = os.path.join(output_dir, "dem", tif_filename)

    window_size_m = 400
    expected_res = 0.25
    threshold = 10
    rock_threshold = 30
    kernel_size = 3

    # Process VMI (canopy) data
    try:
        with rasterio.open(vmi_path) as vmi_src:
            pixel_size = vmi_src.res[0]
            window_size_pixels = int(window_size_m / pixel_size)
            half_window = window_size_pixels // 2
            row_idx, col_idx = vmi_src.index(row['x'], row['y'])

            col_start = max(0, col_idx - half_window)
            row_start = max(0, row_idx - half_window)
            col_end = min(vmi_src.width, col_idx + half_window)
            row_end = min(vmi_src.height, row_idx + half_window)

            if col_end <= col_start or row_end <= row_start:
                return features

            window = rasterio.windows.Window(col_start, row_start, col_end - col_start, row_end - row_start)
            canopy = vmi_src.read(1, window=window, masked=True)
            valid_canopy = canopy[canopy != 32767]
            features['avg_canopy_cover'] = float(np.mean(valid_canopy))
            features['std_canopy_cover'] = float(np.std(valid_canopy))
            canopy_data = canopy.filled(32767)  # Replace masked values
            forest_mask = (canopy_data >= threshold) & (canopy_data != 32767)
            features['prop_forested_area'] = float(np.sum(forest_mask) / forest_mask.size)
            labeled, num_patches = label(forest_mask)
            features['num_forest_patches'] = num_patches
            edge_mask = forest_mask ^ binary_erosion(forest_mask, structure=np.ones((kernel_size, kernel_size), dtype=bool))
            features['edge_density'] = float(np.sum(edge_mask) / forest_mask.size)
    except:
        pass

    # Process DTW data with improved masking
    try:
        with rasterio.open(dtw_path) as dtw_src:
            pixel_size = dtw_src.res[0]
            window_size_pixels = int(window_size_m / pixel_size)
            half_window = window_size_pixels // 2
            row_idx, col_idx = dtw_src.index(row['x'], row['y'])

            col_start = max(0, col_idx - half_window)
            row_start = max(0, row_idx - half_window)
            col_end = min(dtw_src.width, col_idx + half_window)
            row_end = min(dtw_src.height, row_idx + half_window)

            if col_end <= col_start or row_end <= row_start:
                return features

            window = rasterio.windows.Window(col_start, row_start, col_end - col_start, row_end - row_start)
            dtw = dtw_src.read(1, window=window, masked=True)

            # Use raster's nodata value if defined, otherwise mask known invalid values
            if dtw_src.nodata is not None:
                dtw_data = np.ma.masked_equal(dtw, dtw_src.nodata)
                logger.debug(f"Using nodata value from metadata: {dtw_src.nodata}")
            else:
                dtw_data = np.ma.masked_where(np.isin(dtw, [32767, -32768, 65535]), dtw)
                logger.debug("No nodata value in metadata; masking 32767, -32768 and 65535")

            valid_mask = ~dtw_data.mask
            if np.any(valid_mask):
                valid_dtw = dtw_data[valid_mask].astype(np.float32)
                logger.debug(f"Valid DTW pixels: {valid_dtw.size}, range: {valid_dtw.min()} to {valid_dtw.max()}")
                wetland_count = np.count_nonzero((valid_dtw < 1) & ~np.isnan(valid_dtw))
                valid_count = np.count_nonzero(~np.isnan(valid_dtw))
                features['prop_wetland_area'] = float(wetland_count / valid_count)
                features['avg_dtw'] = float(valid_dtw.mean())
                features['std_dtw'] = float(valid_dtw.std())
            else:
                logger.debug("No valid DTW pixels in window")
                features['prop_wetland_area'] = np.nan
                features['avg_dtw'] = np.nan
                features['std_dtw'] = np.nan
    except:
        pass

    # Process DEM (elevation) data
    try:
        with rasterio.open(dem_path) as dem_src:
            pixel_size = dem_src.res[0]
            window_size_pixels = int(window_size_m / pixel_size)
            half_window = window_size_pixels // 2
            row_idx, col_idx = dem_src.index(row['x'], row['y'])

            col_start = max(0, col_idx - half_window)
            row_start = max(0, row_idx - half_window)
            col_end = min(dem_src.width, col_idx + half_window)
            row_end = min(dem_src.height, row_idx + half_window)

            if col_end <= col_start or row_end <= row_start:
                return features

            window = rasterio.windows.Window(col_start, row_start, col_end - col_start, row_end - row_start)
            dem = dem_src.read(1, window=window, masked=True)
            dem_data = dem.filled(np.nan)
            features['avg_elevation'] = float(np.nanmean(dem_data))

            dy, dx = np.gradient(dem_data, pixel_size)
            slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
            features['avg_slope'] = float(np.nanmean(slope))
            features['std_slope'] = float(np.nanstd(slope))
            features['prop_rocky_outcrops'] = float(np.sum(slope > rock_threshold) / slope.size)
    except:
        pass

    return features


# --- New function for tree-level features ---
def compute_tree_level_features(row, output_dir):
    """
    Compute tree-level features for a single row.
    This function computes the distances to forest edge, wetland, and rocky outcrop,
    and gathers contextual statistics using compute_additional_features.
    """
    results = {}

    tif_filename = row['Filename']
    if pd.isna(tif_filename):
        logger.debug("No TIFF filename recorded for tree row. Skipping distance calculations.")
        # Still attempt to compute additional features (will return NaNs)
    else:
        vmi_path = os.path.join(output_dir, "vmi", tif_filename)
        dem_path = os.path.join(output_dir, "dem", tif_filename)
        dtw_path = os.path.join(output_dir, "dtw", tif_filename)

        # Forest edge distance
        if not (os.path.exists(vmi_path) and os.path.exists(dem_path)):
            logger.warning(f"TIFF files missing for forest edge at {vmi_path} or {dem_path}.")
            results['distance_to_forest_edge'] = None
        else:
            try:
                results['distance_to_forest_edge'] = distance_to_forest_edge(
                    row['x'], row['y'], vmi_path, dem_path
                )
            except Exception as e:
                logger.error(f"Error computing forest edge distance for {tif_filename}: {e}")
                results['distance_to_forest_edge'] = None

        # Wetland distance
        if not os.path.exists(dtw_path):
            logger.warning(f"TIFF file for wetland not found at {dtw_path}.")
            results['distance_to_nearest_wetland'] = None
        else:
            try:
                results['distance_to_nearest_wetland'] = distance_to_nearest_wetland(
                    dtw_path, row['x'], row['y']
                )
            except Exception as e:
                logger.error(f"Error computing wetland distance for {tif_filename}: {e}")
                results['distance_to_nearest_wetland'] = None

        # Rocky outcrop distance
        if not os.path.exists(dem_path):
            logger.warning(f"TIFF file for rocky outcrop not found at {dem_path}.")
            results['distance_to_rocky_outcrop'] = None
        else:
            try:
                results['distance_to_rocky_outcrop'] = distance_to_rocky_outcrop(
                    dem_path, row['x'], row['y']
                )
            except Exception as e:
                logger.error(f"Error computing rocky outcrop distance for {tif_filename}: {e}")
                results['distance_to_rocky_outcrop'] = None

    # Add contextual features
    additional = compute_additional_features(row, output_dir)
    results.update(additional)
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
    logger.debug(f"Columns in computed distances: {distances_df.columns.tolist()}")
    merged_df = pd.concat([merged_df, distances_df], axis=1)
    logger.debug(f"Final merged DataFrame columns: {merged_df.columns.tolist()}")
    return merged_df


def load_cluster_data(data_path):
    cluster_df = pd.read_csv(data_path)

    def generate_patch_id(row):
        unique_str = f"{row['x']}_{row['y']}"
        return hashlib.md5(unique_str.encode()).hexdigest()

    cluster_df["patch_id"] = cluster_df.apply(generate_patch_id, axis=1)
    return cluster_df


import geopandas as gpd
import pandas as pd
import numpy as np

def load_coords(data_path):
    # Load the GeoPackage file
    data = gpd.read_file(data_path)

    # Check Coordinate Reference System (CRS) and reproject if necessary
    if data.crs.is_geographic:
        # Example: Reproject to UTM Zone 10N (adjust based on data location)
        data = data.to_crs('EPSG:32610')

    # Extract coordinates for clustering
    coords = pd.DataFrame({'x': data.geometry.x, 'y': data.geometry.y})

    return coords


def main():
    parser = argparse.ArgumentParser(description="Process tree cluster data and extract patches.")
    parser.add_argument("--data-path", required=True, help="Path to the input data file (e.g., .gpkg or .csv).")
    parser.add_argument("--output-dir", required=True, help="Directory to save the output patches and mapping file.")
    args = parser.parse_args()

    data_path = args.data_path
    output_dir = args.output_dir

    collection_names = ["dtw", "dem", "vmi"]
    mapping_csv_path = os.path.join(output_dir, "mapping.csv")

    coords = load_coords(data_path)
    
    


if __name__ == "__main__":
    main()

'''

Usage:

python ./src/distances.py --data-path ./output/clusters.csv --output-dir ./output

sbatch ~/TreeClusters/scripts/run_distances.sh lumi

scp rahmanan@lumi.csc.fi:/scratch/project_462000684/rahmanan/tree_clusters/output/clusters_with_distance.csv ~/Documents/TreeClusters/output

'''

def add_features_to_tree_df(tree_df, mapping_csv_path, output_dir, num_workers=None):
    """
    Adds distance and contextual features to a tree-level DataFrame.
    Loads mapping.csv, merges Filename into tree_df using TreeID, computes features in parallel, and returns enhanced DataFrame.
    """
    # Load mapping.csv
    mappings = pd.read_csv(mapping_csv_path)
    # Merge Filename into tree_df using TreeID or similar unique identifier
    if 'TreeID' in mappings.columns:
        merge_col = 'TreeID'
    elif 'patch_id' in mappings.columns:
        merge_col = 'patch_id'
    else:
        raise ValueError("mapping.csv must contain 'TreeID' or 'patch_id' column")
    # Try to find the matching column in tree_df
    if merge_col not in tree_df.columns:
        # Try to find 'TreeID' or 'patch_id' in tree_df
        if 'TreeID' in tree_df.columns:
            merge_col_df = 'TreeID'
        elif 'patch_id' in tree_df.columns:
            merge_col_df = 'patch_id'
        else:
            raise ValueError("tree_df must contain 'TreeID' or 'patch_id' column")
    else:
        merge_col_df = merge_col
    merged_df = pd.merge(tree_df, mappings[[merge_col, 'Filename']], left_on=merge_col_df, right_on=merge_col, how='left')

    if num_workers is None:
        num_workers = cpu_count()
    logger.info(f"Using {num_workers} CPU workers for parallel tree-level feature computation.")

    rows = [row for _, row in merged_df.iterrows()]
    args = [(row, output_dir) for row in rows]
    # Use Pool for parallelization
    with Pool(processes=num_workers) as pool:
        features_list = list(tqdm(pool.imap(lambda args: compute_tree_level_features(*args), args), total=len(merged_df), desc="Computing tree-level features"))

    features_df = pd.DataFrame(features_list, index=merged_df.index)
    enhanced_df = pd.concat([merged_df, features_df], axis=1)
    return enhanced_df