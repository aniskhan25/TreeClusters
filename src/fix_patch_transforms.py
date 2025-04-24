#!/usr/bin/env python3
import os
import argparse
import rasterio
from rasterio.transform import from_bounds
import pandas as pd

def fix_transforms(mapping_csv, patches_root, patch_size, resolution):
    """
    For each entry in mapping_csv, locate the corresponding TIFF under patches_root,
    recompute its geotransform so it’s centered on (X,Y), and overwrite it in place.
    """
    # Compute half the geographic extent of a patch
    min_extent = 300.0
    default_extent = patch_size * resolution
    extent = max(default_extent, min_extent)
    half_size = extent / 2.0

    # Read the CSV that lists: Filename, X, Y
    df = pd.read_csv(mapping_csv)

    for _, row in df.iterrows():
        fname = row["Filename"]
        x_center, y_center = row["X"], row["Y"]

        # Recursively search for this file under patches_root
        found = False
        for root, _, files in os.walk(patches_root):
            if fname in files:
                found = True
                path = os.path.join(root, fname)
                print(f"Processing {path}…")

                # Open, read data & profile
                with rasterio.open(path) as src:
                    data = src.read()
                    prof = src.profile

                # Compute new transform
                new_tf = from_bounds(
                    x_center - half_size,
                    y_center - half_size,
                    x_center + half_size,
                    y_center + half_size,
                    prof["width"],
                    prof["height"]
                )

                # Update profile and overwrite
                prof.update(transform=new_tf)
                with rasterio.open(path, "w", **prof) as dst:
                    dst.write(data)

                print(f"  → Rewrote transform centered at ({x_center}, {y_center})")
                break

        if not found:
            print(f"Warning: {fname} not found under {patches_root}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Fix geotransform on existing patch TIFFs based on mapping.csv"
    )
    p.add_argument(
        "--mapping-csv", required=True,
        help="Path to mapping.csv (with columns Filename, X, Y)"
    )
    p.add_argument(
        "--patches-root", required=True,
        help="Root directory containing your per-collection subfolders of TIFFs"
    )
    p.add_argument(
        "--patch-size", type=int, default=1200,
        help="patch_size used originally (in pixels)"
    )
    p.add_argument(
        "--resolution", type=float, default=0.25,
        help="resolution used originally (in meters per pixel)"
    )
    args = p.parse_args()

    fix_transforms(
        mapping_csv=args.mapping_csv,
        patches_root=args.patches_root,
        patch_size=args.patch_size,
        resolution=args.resolution
    )


    '''
    Usage:

    python ./src/fix_patch_transforms.py \
        --mapping-csv output/mapping.csv \
        --patches-root output \
        --patch-size 1200 \
        --resolution 0.25
        
    '''