import math
from rasterio.windows import Window
import rasterio
import numpy as np
from pathlib import Path
from tqdm import tqdm

PATCH_SIZE = 512

# === Paths ===
cwd = Path(__file__).resolve().parent
data_dir = cwd.parent / "data"
processed_dir = data_dir / "processed"
bands_dir = processed_dir / "merged_bands"
ref_path = processed_dir / "GBDA24_ex2_ref_data_reprojected.tif"

# Output patch dataset dir
dataset_dir = data_dir / "patch_dataset"
image_dir = dataset_dir / "images"
label_dir = dataset_dir / "labels"
mask_dir = dataset_dir / "masks"

if __name__ == "__main__":
    for d in [image_dir, label_dir, mask_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # === Open label for shape/metadata ===
    with rasterio.open(ref_path) as ref:
        H, W = ref.height, ref.width
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_nodata = ref.nodata
        ref_meta = ref.meta.copy()

    # === Prepare band file handles ===
    band_ids = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
                "B08", "B8A", "B09", "B10", "B11", "B12"]
    band_files = {b: rasterio.open(bands_dir / f"{b}_merged.tif") for b in band_ids}

    # === Prepare label file handle ===
    ref = rasterio.open(ref_path)
    label_colormap = ref.colormap(1) if ref.count == 1 and ref.colorinterp[0].name == "palette" else None

    stride = PATCH_SIZE
    patch_id = 0

    with rasterio.open(ref_path) as src:
        data = src.read(1)
        nodata = src.nodata
        height, width = src.height, src.width
    valid_mask = data != nodata
    patch_count = 0
    row_count = 0
    col_count = 0
    for row_idx, row in enumerate(range(0, height, PATCH_SIZE)):
        col_valid = 0
        for col_idx, col in enumerate(range(0, width, PATCH_SIZE)):
            window = Window(col, row,
                            min(PATCH_SIZE, width - col),
                            min(PATCH_SIZE, height - row))
            patch = valid_mask[
                int(window.row_off):int(window.row_off + window.height),
                int(window.col_off):int(window.col_off + window.width)
            ]
            if patch.any():
                patch_count += 1
                col_valid += 1
        if col_valid > 0:
            row_count += 1
            col_count = max(col_count, col_valid)
    print(f"Total valid patches: {patch_count}")
    print(f"Grid size: {row_count} rows × {col_count} cols")
    num_digits = len(str(patch_count))

    row_steps = list(range(0, H, stride))
    col_steps = list(range(0, W, stride))
    print("Creating patches (streamed):")
    for row_idx, row in enumerate(tqdm(row_steps, desc="Rows", colour="red")):
        for col_idx, col in enumerate(tqdm(col_steps, desc=f"Cols", leave=False, colour="blue")):
            window = rasterio.windows.Window(
                col_off=col,
                row_off=row,
                width=min(PATCH_SIZE, W - col),
                height=min(PATCH_SIZE, H - row)
            )

            # === Read label + mask
            lbl_patch = ref.read(1, window=window)
            msk_patch = (lbl_patch != ref_nodata)

            if not msk_patch.any():
                continue  # all padding, skip

            # === Read band data for this window
            band_data = []
            for b in band_ids:
                band = band_files[b]
                arr = band.read(1, window=window)
                band_data.append(arr)

            img_patch = np.stack(band_data, axis=0)

            # === Pad if needed
            pad_h = PATCH_SIZE - img_patch.shape[1]
            pad_w = PATCH_SIZE - img_patch.shape[2]

            if pad_h > 0 or pad_w > 0:
                img_patch = np.pad(img_patch, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=0)
                lbl_patch = np.pad(lbl_patch, ((0, pad_h), (0, pad_w)), constant_values=0)
                msk_patch = np.pad(msk_patch, ((0, pad_h), (0, pad_w)), constant_values=0)

            # === Save patch
            patch_meta = ref_meta.copy()
            patch_meta.update({
                "height": PATCH_SIZE,
                "width": PATCH_SIZE,
                "transform": rasterio.windows.transform(window, ref_transform),
                "crs": ref_crs
            })

            # Save image
            patch_meta.update(count=len(band_ids), dtype=np.uint16)
            img_path = image_dir / f"image_{patch_id:0{num_digits}d}.tif"
            with rasterio.open(img_path, "w", **patch_meta) as dst:
                dst.write(img_patch.astype(np.uint16))

            # Save label
            patch_meta.update(count=1, dtype=lbl_patch.dtype)
            lbl_path = label_dir / f"label_{patch_id:0{num_digits}d}.tif"
            with rasterio.open(lbl_path, "w", **patch_meta) as dst:
                dst.write(lbl_patch, 1)
                if label_colormap:
                    dst.write_colormap(1, label_colormap)

            # Save mask
            msk_path = mask_dir / f"mask_{patch_id:0{num_digits}d}.tif"
            with rasterio.open(msk_path, "w", **patch_meta) as dst:
                dst.write(msk_patch.astype(np.uint8), 1)

            patch_id += 1

    # === Cleanup ===
    ref.close()
    for b in band_files.values():
        b.close()

    print(f"{patch_id} patches created.")

