from pathlib import Path
import rasterio
from rasterio.merge import merge
from rasterio.windows import from_bounds
from rasterio.windows import transform as window_transform
from contextlib import ExitStack
import numpy as np
from tqdm import tqdm
import gc

# === Paths ===
cwd = Path(__file__).resolve().parent
data_dir = cwd.parent / "data"
processed_dir = data_dir / "processed"
ref_path = processed_dir / "GBDA24_ex2_ref_data_reprojected.tif"

# === Bands to merge ===
band_names = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
              "B08", "B8A", "B09", "B10", "B11", "B12"]

tile_dirs = [d for d in processed_dir.glob("S2A_MSIL1C_*") if d.is_dir()]

# === Get valid bounds of reference (non-nodata area) ===
with rasterio.open(ref_path) as ref_src:
    ref_crs = ref_src.crs
    ref_transform = ref_src.transform
    ref_profile = ref_src.profile
    nodata = ref_src.nodata

    full_data = ref_src.read(1)
    mask = full_data != nodata
    if not mask.any():
        raise RuntimeError("Reference image has no valid data.")

    rows, cols = np.where(mask)
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    valid_window = rasterio.windows.Window(
        col_off=col_min,
        row_off=row_min,
        width=col_max - col_min + 1,
        height=row_max - row_min + 1
    )
    ref_bounds = rasterio.windows.bounds(valid_window, ref_transform)

# === Output dir ===
merged_out_dir = processed_dir / "merged_bands"
merged_out_dir.mkdir(exist_ok=True)

# === Process each band ===
for band in tqdm(band_names, desc="Merging Bands", colour="blue"):
    band_paths = [p for d in tile_dirs for p in d.glob(f"{band}_10m.tif")]
    if not band_paths:
        print(f"[WARN] No data found for {band}")
        continue

    with ExitStack() as stack:
        srcs = [stack.enter_context(rasterio.open(p)) for p in band_paths]
        merged, merged_transform = merge(srcs)
        profile = srcs[0].profile.copy()

    # Write full merged image without cropping
    out_path = merged_out_dir / f"{band}_merged.tif"
    profile.update({
        "height": merged.shape[1],
        "width": merged.shape[2],
        "transform": merged_transform,
        "dtype": rasterio.uint16,
        "count": 1
    })

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(merged[0].astype(np.uint16), 1)

    # # Crop to valid ref bounds
    # crop_window = from_bounds(*ref_bounds, transform=merged_transform)
    # crop_window = crop_window.round_offsets().round_lengths()
    # cropped = merged[:,
    #     crop_window.row_off:crop_window.row_off + crop_window.height,
    #     crop_window.col_off:crop_window.col_off + crop_window.width
    # ]
    #
    # cropped_transform = window_transform(crop_window, merged_transform)
    #
    # # Write to single output .tif
    # out_path = merged_out_dir / f"{band}_merged_cropped.tif"
    # profile.update({
    #     "height": cropped.shape[1],
    #     "width": cropped.shape[2],
    #     "transform": cropped_transform,
    #     "dtype": rasterio.uint16,
    #     "count": 1
    # })

    # with rasterio.open(out_path, "w", **profile) as dst:
    #     dst.write(cropped[0].astype(np.uint16), 1)

    for src in srcs:
        if not src.closed:
            src.close()

    # === Free memory ===
    # del merged, cropped, merged_transform, crop_window, profile, srcs
    del merged, merged_transform, profile, srcs
    gc.collect()

print("All bands merged and cropped to match reference label area.")
