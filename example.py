import os

from compute_distances import (
    distance_to_forest_edge,
    distance_to_rocky_outcrop,
    distance_to_nearest_wetland,
)


def main():
    patch_id = "00064254"
    output_dir = "./output"

    vmi_path = os.path.join(output_dir, "vmi", patch_id + ".tif")
    dem_path = os.path.join(output_dir, "dem", patch_id + ".tif")
    dtw_path = os.path.join(output_dir, "dtw", patch_id + ".tif")

    x, y = 431605.61,7080224.15

    distance = distance_to_forest_edge(x, y, vmi_path, dem_path, window_size=20)
    print(f"Distance from the point to the forest edge: {distance:.2f} meters.")

    distance = distance_to_rocky_outcrop(dem_path, x, y)
    print(f"Distance to nearest rocky outcrop: {distance:.2f} meters")

    distance = distance_to_nearest_wetland(dtw_path, x, y)
    print(f"Distance to the nearest wetland: {distance:.2f} meters")


if __name__ == "__main__":
    main()
