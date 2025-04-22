import os
import csv
import time
import logging
import hashlib
import rasterio
import argparse
import threading
import pystac_client

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm import tqdm

from affine import Affine
from rasterio.warp import transform
from rasterio.windows import Window
from rasterio.transform import from_bounds
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from rtree import index as rtree_index
from shapely.geometry import Point, box


class PatchProcessor:
    def __init__(
        self,
        data_path,
        catalog_url,
        patch_size=256,
        buffer=5000,
        resolution=0.5,
        epsg=3067,
        offset=3,
        # safe_margin=25,  # if you want a 50m buffer, or computed as (patch_size/2) * resolution
    ):
        self.data_path = data_path
        self.catalog_url = catalog_url
        self.patch_size = patch_size
        self.buffer = buffer
        self.resolution = resolution
        self.epsg = epsg
        self.cluster_df = None
        self.catalog = None
        self.mapping_file = None
        self.offset = offset

        # Initialize spatial index and patch mapping for greedy assignment.
        self.patch_index = rtree_index.Index()
        self.extracted_patch_mapping = {}  # key: patch_id, value: (footprint, patch_filepath, patch_id)
        self.patch_id_counter = 0

        self.lock = threading.Lock()

    def load_cluster_data(self, num_samples=None, source_type="csv", gpkg_layer=None):
        if source_type == "csv":
            self.cluster_df = pd.read_csv(self.data_path)
        elif source_type == "gpkg":
            gdf = gpd.read_file(self.data_path, layer=gpkg_layer) if gpkg_layer else gpd.read_file(self.data_path)
            gdf = gdf.to_crs(epsg=self.epsg)
            gdf["x"] = gdf.geometry.x
            gdf["y"] = gdf.geometry.y

            # Compute cluster IDs based on offset grid
            gdf["cluster_id"] = (
                np.floor(gdf["x"] / (self.offset * self.resolution)).astype(int).astype(str) + "_" +
                np.floor(gdf["y"] / (self.offset * self.resolution)).astype(int).astype(str)
            )

            self.cluster_df = gdf.drop(columns="geometry")
        else:
            raise ValueError("Unsupported source_type. Use 'csv' or 'gpkg'.")

        def generate_patch_id(row):
            unique_str = f"{row['x']}_{row['y']}"
            return hashlib.md5(unique_str.encode()).hexdigest()[:8]

        def add_patch_ids(df):
            df["patch_id"] = df.apply(generate_patch_id, axis=1)
            return df

        self.cluster_df = add_patch_ids(self.cluster_df)

        if num_samples:
            self.cluster_df = self.cluster_df.sample(n=num_samples, random_state=7).reset_index(drop=True)

        logger.debug("Survey data loaded successfully.")

    def connect_to_catalog_with_retry(self, retries=3, delay=5):
        for attempt in range(retries):
            try:
                self.catalog = pystac_client.Client.open(self.catalog_url)
                logger.debug(f"Connected to catalog: {self.catalog.title}")
                return
            except pystac_client.exceptions.APIError as e:
                logger.error(f"Connection failed: {e}. Retrying {retries - attempt - 1} more times...")
                time.sleep(delay)
        raise ConnectionError(f"Failed to connect to STAC catalog after {retries} retries.")

    def search_catalog(self, lon, lat, collections_info):
        results = []
        try:
            for info in collections_info.values():
                search = self.catalog.search(
                    intersects={"type": "Point", "coordinates": [lon, lat]},
                    collections=[info['collection']],
                    datetime=f"{info['year_start']}-01-01/{info['year_end']}-12-31",
                )
                items = search.item_collection()

                if items:
                    results.append(items[0])  # Assume the first item is relevant
                    logger.debug(f"Found item in collection {info['collection']}.")
                else:
                    logger.debug(f"No items found in collection {info['collection']}.")
        except pystac_client.exceptions.APIError as e:
            logger.error(f"STAC query failed for point ({lon}, {lat}): {e}")
        return results

    # def extract_patch(self, lon, lat, raster_url):
    #     print(lon, lat, raster_url)
    #     with rasterio.open(raster_url) as src:
    #         lon_lat_crs = "EPSG:4326"  # WGS84
    #         raster_crs = src.crs
    #         x, y = rasterio.warp.transform(lon_lat_crs, raster_crs, [lon], [lat])
    #         x, y = x[0], y[0]

    #         print(x,y)

    #         py, px = src.index(x, y)

    #         px = max(0, min(px, src.width - 1))
    #         py = max(0, min(py, src.height - 1))

    #         print(px, py)

    #         window = Window(
    #             col_off=max(0, px - self.patch_size // 2),
    #             row_off=max(0, py - self.patch_size // 2),
    #             width=min(self.patch_size, src.width - px + self.patch_size // 2),
    #             height=min(self.patch_size, src.height - py + self.patch_size // 2),
    #         )

    #         print(window)

    #         patch = src.read(window=window)
    #         transform = src.window_transform(window)

    #         scale_factor_x = (src.res[0] / self.resolution)  # src.res[0] is the original x resolution
    #         scale_factor_y = (src.res[1] / self.resolution)  # src.res[1] is the original y resolution

    #         print(src.res[0], src.res[1], self.resolution)
    #         print(scale_factor_x, scale_factor_y)

    #         new_width = int(patch.shape[2] * scale_factor_x)
    #         new_height = int(patch.shape[1] * scale_factor_y)
    #         new_transform = transform * Affine.scale(1 / scale_factor_x, 1 / scale_factor_y)

    #         resampled_patch = np.empty((patch.shape[0], new_height, new_width), dtype=patch.dtype)

    #         print(new_width, new_height)

    #         for i in range(patch.shape[0]):
    #             resampled_patch[i], _ = rasterio.warp.reproject(
    #                 source=patch[i],
    #                 destination=resampled_patch[i],
    #                 src_transform=transform,
    #                 dst_transform=new_transform,
    #                 src_crs=src.crs,
    #                 dst_crs=src.crs,
    #                 resampling=rasterio.enums.Resampling.bilinear,
    #             )

    #         return resampled_patch, new_transform, src.crs

    def extract_patch(self, lon, lat, raster_url):
        min_extent = 300.0  # Minimum geographic width (in meters) for the patch
        default_extent = self.patch_size * self.resolution
        extent = max(default_extent, min_extent)
        half_size = extent / 2.0

        with rasterio.open(raster_url) as src:
            x_list, y_list = rasterio.warp.transform("EPSG:4326", src.crs, [lon], [lat])
            x_center, y_center = x_list[0], y_list[0]

            left = x_center - half_size
            bottom = y_center - half_size
            right = x_center + half_size
            top = y_center + half_size

            # Remove deprecated width and height parameters
            window = rasterio.windows.from_bounds(left, bottom, right, top, src.transform)

            patch = src.read(window=window)
            new_transform = src.window_transform(window)

            return patch, new_transform, src.crs

    def save_patch_as_tiff(self, patch, transform, crs, patch_filepath, band_descriptions):
        profile = {
            "driver": "GTiff",
            "height": patch.shape[1],
            "width": patch.shape[2],
            "count": patch.shape[0],
            "dtype": patch.dtype,
            "crs": crs,
            "transform": transform,
        }

        with rasterio.open(patch_filepath, "w", **profile) as dst:
            dst.write(patch)
            for i, description in enumerate(band_descriptions, start=1):
                dst.set_band_description(i, description)

        logger.debug(f"Patch saved to {patch_filepath} with band descriptions: {band_descriptions}")

    def create_extended_patch(self, lon, lat, items, patch_id, output_dir, collection_names):
        for idx, item in enumerate(items):
            asset_keys = list(item.assets.keys())
            if len(asset_keys) < 1:
                logger.debug(f"No assets found for item {item.id}. Skipping.")
                continue

            logger.debug(f"Processing item {item.id} with assets: {asset_keys}")

            patch, transform, crs = self.extract_patch(lon, lat, item.assets[asset_keys[0]].href)

            patch_filepath = os.path.join(output_dir, collection_names[idx], patch_id + '.tif')

            self.save_patch_as_tiff(
                patch,
                transform,
                crs,
                patch_filepath,
                [f"Band_{i+1}" for i in range(patch.shape[0])],
            )

    def plot_patch(self, patch):
        norm_patch = (patch - np.nanmin(patch)) / (np.nanmax(patch) - np.nanmin(patch))
        plt.figure(figsize=(8, 8))
        plt.imshow(norm_patch.transpose(1, 2, 0))  # Assuming multi-band raster
        plt.title("Patch Centered at Specified Lat/Lon")
        plt.axis("off")
        plt.show()

    def process_survey_point(self, idx, output_dir, collections_info):
        collection_names = list(collections_info.keys())

        if self.cluster_df is None:
            raise ValueError("Survey data has not been loaded.")

        x = self.cluster_df["x"].iloc[idx]
        y = self.cluster_df["y"].iloc[idx]
        patch_id = self.cluster_df["patch_id"].iloc[idx]

        if pd.isna(x) or pd.isna(y):
            logger.debug(f"Skipping survey point {idx} due to missing data: x={x}, y={y}")
            return

        logger.debug(f"Processing survey point {idx}: x={x}, y={y}")

        lon, lat = transform(f"EPSG:{self.epsg}", "EPSG:4326", [x], [y])
        lon, lat = lon[0], lat[0]

        # x_list, y_list = transform("EPSG:4326", f"EPSG:{self.epsg}", [x], [y])
        # x, y = x_list[0], y_list[0]
        point_geom = Point(x, y)

        safe_margin = (self.patch_size / 2) * self.resolution

        point_bbox = (
            x - safe_margin,
            y - safe_margin,
            x + safe_margin,
            y + safe_margin,
        )

        assigned_patch_filename = None

        with self.lock:
            for numeric_patch_id in self.patch_index.intersection(point_bbox):
                footprint, patch_filename, existing_patch_id = self.extracted_patch_mapping[numeric_patch_id]
                minx, miny, maxx, maxy = footprint.bounds

                if (
                    x - minx >= safe_margin
                    and maxx - x >= safe_margin
                    and y - miny >= safe_margin
                    and maxy - y >= safe_margin
                ):
                    assigned_patch_id = existing_patch_id
                    assigned_patch_filename = patch_filename
                    logger.debug(
                        f"Survey point {existing_patch_id} assigned to existing patch {os.path.basename(patch_filename)} with safe margin."
                    )
                    break

        if assigned_patch_filename is not None:
            self.record_mapping(
                patch_id,
                assigned_patch_filename,
                assigned_patch_id,
                x,
                y,
            )
            return

        items = self.search_catalog(lon, lat, collections_info)
        if len(items) < len(collection_names):
            logger.debug(f"Skipping survey point {idx} due to insufficient data from collections.")
            return

        try:
            self.create_extended_patch(lon, lat, items, patch_id, output_dir, collection_names)
            self.record_mapping(patch_id, patch_id + '.tif', patch_id, x, y)
            logger.debug(f"Extracted patches for survey point {patch_id}.")

            patch_extent = self.patch_size * self.resolution
            patch_footprint = box(x - patch_extent, y - patch_extent, x + patch_extent, y + patch_extent)

            with self.lock:
                numeric_patch_id = self.patch_id_counter
                self.patch_id_counter += 1
                self.patch_index.insert(numeric_patch_id, patch_footprint.bounds)
                self.extracted_patch_mapping[numeric_patch_id] = (
                    patch_footprint,
                    patch_id + '.tif',
                    patch_id,
                )
        except Exception as e:
            logger.error(f"Failed to process survey point {idx} due to error: {e}")

    def record_mapping(self, sequence_num, patch_file, survey_idx, x, y):
        if not self.mapping_file:
            raise ValueError("Mapping file path is not set.")

        with self.lock:
            file_exists = os.path.isfile(self.mapping_file)

            with open(self.mapping_file, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(
                        [
                            "TreeID",
                            "Filename",
                            "PatchID",
                            "X",
                            "Y",
                        ]
                    )
                writer.writerow([sequence_num, patch_file, survey_idx, x, y])

            logger.debug(f"Metadata recorded: TreeID={sequence_num}, Filename={patch_file}")

    def setup_mapping_file(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.mapping_file = os.path.join(output_dir, "mapping.csv")
        if os.path.isfile(self.mapping_file):
            self.processed_sequences = pd.read_csv(self.mapping_file)["TreeID"].tolist()
            logger.debug(f"Loaded {len(self.processed_sequences)} processed sequences from mapping file.")
        else:
            self.processed_sequences = []
            logger.debug(f"No existing mapping file found. A new one will be created at {self.mapping_file}.")


def main():
    parser = argparse.ArgumentParser(description="Process tree cluster data and extract patches.")
    parser.add_argument("--data-path", required=True, help="Path to the input data file (e.g., .gpkg or .csv).")
    parser.add_argument("--output-dir", required=True, help="Directory to save the output patches and mapping file.")
    args = parser.parse_args()

    data_path = args.data_path
    output_dir = args.output_dir

    catalog_url = "https://paituli.csc.fi/geoserver/ogc/stac/v1"
    patch_size = 1200
    buffer = 5000
    resolution = 0.25  # in meters
    epsg = 3067

    collections_info = {
        "dtw": {
            "collection": "luke_dtw_2m_0_5ha_threshold_at_paituli",
            "year_start": 2019,
            "year_end": 2019,
        },
        "dem": {
            "collection": "nls_digital_elevation_model_2m_at_paituli",
            "year_start": 2008,
            "year_end": 2020,
        },
        "vmi": {
            "collection": "luke_vmi_latvuspeitto_at_paituli",
            "year_start": 2021,
            "year_end": 2021,
        },
    }

    # collections = [info["collection"] for info in collections_info.values()]
    collection_names = list(collections_info.keys())

    processor = PatchProcessor(
        data_path=data_path,
        catalog_url=catalog_url,
        patch_size=patch_size,
        buffer=buffer,
        resolution=resolution,
        epsg=epsg,
    )

    processor.load_cluster_data(source_type="gpkg")
    # processor.load_cluster_data(num_samples=1)  # USE ONLY FOR DEBUGGING

    processor.connect_to_catalog_with_retry()

    processor.setup_mapping_file(output_dir)

    for name in collection_names:
        patches_dir = os.path.join(output_dir, name)
        os.makedirs(patches_dir, exist_ok=True)

    def process_cluster_row(idx, row):
        try:
            patch_id = row["patch_id"]
            patch_filename = f"{patch_id}.tif"

            # Check if any of the patch files already exist
            patch_exists = False
            for name in collection_names:
                patch_filepath = os.path.join(output_dir, name, patch_filename)
                if os.path.exists(patch_filepath):
                    logger.debug(f"File {patch_filename} already exists. Skipping survey point {patch_id}.")
                    patch_exists = True
                    break  # Stop checking further if one file is found

            if not patch_exists:
                # Process the survey point since no file exists yet
                processor.process_survey_point(idx, output_dir, collections_info)
            else:
                logger.debug(f"Skipping processing for survey point {patch_id} because file already exists.")

        except Exception as e:
            logger.error(f"Error processing survey point {patch_id}: {e}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        total_rows = len(processor.cluster_df)
        for idx, row in processor.cluster_df.iterrows():
            logger.debug(f"Submitting survey point {idx} out of {total_rows} ...")
            future = executor.submit(process_cluster_row, idx, row)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=total_rows, desc="Processing survey points"):
            try:
                future.result()
            except Exception as exc:
                logger.error(f"Exception occurred during parallel processing: {exc}")


if __name__ == "__main__":
    main()


'''

Usage:

python ./src/patches.py --data-path ./data/DeadTrees_2023_Anis_ShapeStudy.gpkg --output-dir ./output_shape

sbatch ~/TreeClusters/scripts/run_patches.sh lumi finland

'''