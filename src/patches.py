import os
import csv
import time
import logging
import hashlib
import rasterio
from rasterio.warp import transform
import argparse
import threading
import pystac_client
from functools import lru_cache

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from rtree import index as rtree_index
from shapely.geometry import Point, box

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def expand_path(path):
    return os.path.expandvars(path)


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
        min_extent=300.0,  # Minimum geographic width (in meters) for the patch
        # safe_margin=25,  # if want a 50m buffer, or computed as (patch_size/2) * resolution
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
        self.min_extent = min_extent

        # Initialize spatial index and patch mapping for greedy assignment.
        self.patch_index = rtree_index.Index()
        self.extracted_patch_mapping = {}  # key: patch_id, value: (footprint, patch_filepath, patch_id)
        self.patch_id_counter = 0

        self.lock = threading.Lock()
        self._transformer_to_wgs84 = None
        self.dataset_paths = None
        self.dataset_extents = [300.0, 100.0, 300.0]  # dtw: 300m, dem: 100m, vmi: 300m
    @staticmethod
    @lru_cache(maxsize=32)
    def open_dataset(href):
        return rasterio.open(href)

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
                np.floor(gdf["x"] / (self.offset * self.resolution)).astype(int).astype(str)
                + "_"
                + np.floor(gdf["y"] / (self.offset * self.resolution)).astype(int).astype(str)
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
        pass

    def search_catalog(self, lon, lat, collections_info):
        pass

    def extract_patch_from_dataset(self, lon, lat, dataset, extent):
        extent = max(extent, self.min_extent)
        half_size = extent / 2.0

        x_list, y_list = transform("EPSG:4326", dataset.crs, [lon], [lat])
        x_center, y_center = x_list[0], y_list[0]

        left = x_center - half_size
        bottom = y_center - half_size
        right = x_center + half_size
        top = y_center + half_size

        window = rasterio.windows.from_bounds(left, bottom, right, top, dataset.transform)
        patch = dataset.read(window=window, boundless=True, fill_value=dataset.nodata)
        new_transform = dataset.window_transform(window)

        return patch, new_transform, dataset.crs

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

    def create_extended_patch(self, lon, lat, dataset_paths, patch_id, output_dir, collection_names):
        for idx, dataset_path in enumerate(dataset_paths):
            with self.open_dataset(dataset_path) as dataset:
                extent = self.dataset_extents[idx]
                patch, transform, crs = self.extract_patch_from_dataset(lon, lat, dataset, extent)
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

        if self._transformer_to_wgs84 is None:
            from pyproj import Transformer
            self._transformer_to_wgs84 = Transformer.from_crs(self.epsg, 4326, always_xy=True)
        logger.debug(f"Transformer initialized: {self._transformer_to_wgs84}")
        lon, lat = self._transformer_to_wgs84.transform(x, y)
        logger.debug(f"Transformed coordinates: lon={lon}, lat={lat}")

        point_geom = Point(x, y)

        # Compute per-dataset safe margins
        safe_margins = [extent / 2.0 for extent in self.dataset_extents]
        max_safe_margin = max(safe_margins)

        point_bbox = (
            x - max_safe_margin,
            y - max_safe_margin,
            x + max_safe_margin,
            y + max_safe_margin,
        )

        logger.debug(f"Point bounding box: {point_bbox}")

        assigned_patch_filename = None

        with self.lock:
            for numeric_patch_id in self.patch_index.intersection(point_bbox):
                footprint, patch_filename, existing_patch_id = self.extracted_patch_mapping[numeric_patch_id]
                minx, miny, maxx, maxy = footprint.bounds

                # Use primary dataset's extent for margin check
                patch_extent = self.dataset_extents[0]  # assume primary dataset (e.g., dtw) defines patch size
                safe_margin = patch_extent / 2.0

                if (
                    x >= minx + safe_margin and
                    x <= maxx - safe_margin and
                    y >= miny + safe_margin and
                    y <= maxy - safe_margin
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

        # Instead of querying STAC, use local dataset_paths
        successful = False
        try:
            self.create_extended_patch(lon, lat, self.dataset_paths, patch_id, output_dir, collection_names)
            successful = True
        except Exception as e:
            logger.error(f"Failed to extract all patches for survey point {idx}: {e}")
        finally:
            self.record_mapping(patch_id, patch_id + '.tif', patch_id, x, y)
            if successful:
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

    patch_size = 1200
    buffer = 5000
    resolution = 0.25  # in meters
    epsg = 3067

    # Define collections_info with metadata
    collections_info = {
        "dtw": {
            "name": "dtw",
            "path": "vrt/dtw_005.vrt",
            "year_start": 2023,
            "year_end": 2023,
            "min_extent": 300.0
        },
        "dem": {
            "name": "dem",
            "path": "vrt/dem_2m.vrt",
            "year_start": 2008,
            "year_end": 2020,
            "min_extent": 200.0
        },
        "vmi": {
            "name": "vmi",
            "path": "vmi/2021/latvuspeitto_vmi1x_1721.tif",
            "year_start": 2021,
            "year_end": 2021,
            "min_extent": 300.0
        }
    }

    collection_names = list(collections_info.keys())

    processor = PatchProcessor(
        data_path=data_path,
        catalog_url=None,
        patch_size=patch_size,
        buffer=buffer,
        resolution=resolution,
        epsg=epsg,
    )

    processor.dataset_paths = [os.path.join(data_path, collections_info[name]["path"]) for name in collection_names]
    processor.dataset_extents = [collections_info[name]["min_extent"] for name in collection_names]

    def _determine_source_type():
        _, file_extension = os.path.splitext(data_path)
        if file_extension.lower() == ".csv":
            return "csv"
        elif file_extension.lower() == ".gpkg":
            return "gpkg"
        else:
            raise ValueError("Unsupported file type. Use a .csv or .gpkg file.")

    source_type = _determine_source_type()

    processor.load_cluster_data(source_type=source_type)
    # processor.load_cluster_data(num_samples=1)  # USE ONLY FOR DEBUGGING

    # No longer connect to catalog

    processor.setup_mapping_file(output_dir)

    for name in collection_names:
        patches_dir = os.path.join(output_dir, name)
        os.makedirs(patches_dir, exist_ok=True)

    existing_patches = set()
    for name in collection_names:
        patches_dir = os.path.join(output_dir, name)
        existing_patches.update(
            os.path.splitext(f)[0] for f in os.listdir(patches_dir) if f.endswith(".tif")
        )

    def process_cluster_row(row_idx, row):
        try:
            patch_id = row.patch_id
            patch_filename = f"{patch_id}.tif"

            patch_exists = patch_id in existing_patches
            if patch_exists:
                logger.debug(f"File {patch_filename} already exists. Skipping survey point {patch_id}.")
            else:
                logger.debug(f"Processing survey point {patch_id} ...")
                processor.process_survey_point(row_idx, output_dir, collections_info)

        except Exception as e:
            logger.error(f"Error processing survey point {patch_id}: {e}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        total_rows = len(processor.cluster_df)
        for row_idx, row in enumerate(processor.cluster_df.itertuples(index=False)):
            logger.debug(f"Submitting survey point {row_idx} out of {total_rows} ...")
            future = executor.submit(process_cluster_row, row_idx, row)
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

module use /appl/local/csc/modulefiles
module load geoconda

gdal_translate "STACIT:\"https://paituli.csc.fi/geoserver/ogc/stac/v1/search?collections=luke_dtw_2m_0_5ha_threshold_at_paituli&datetime=2023-01-01/2023-12-31\":asset=luke_dtw_2m_0_5ha_threshold_at_paituli_tiff" -oo max_items=0 -of VRT dtw_005.vrt
gdal_translate "STACIT:\"https://paituli.csc.fi/geoserver/ogc/stac/v1/search?collections=nls_digital_elevation_model_2m_at_paituli&datetime=2008-01-01/2020-12-31\":asset=nls_digital_elevation_model_2m_at_paituli_tiff" -oo max_items=0 -of VRT dem_2m.vrt

scp ./output/clusters.csv rahmanan@lumi.csc.fi:/scratch/project_462000684/rahmanan/tree_clusters/output

python ./src/patches.py --data-path ./output/sample.csv --output-dir ./output

sbatch ~/TreeClusters/scripts/run_patches.sh lumi 0

scp rahmanan@lumi.csc.fi:/scratch/project_462000684/rahmanan/tree_clusters/output/mapping.csv ./output

'''
