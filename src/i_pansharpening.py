import rasterio
from rasterio.enums import Resampling
import numpy as np
from pathlib import Path
from tqdm import tqdm, trange

DEBUG = False

def load_band(img_dir, band_id):
    path = next(img_dir.glob(f"*_{band_id}.jp2"))
    with rasterio.open(path) as src:
        return src.read(1), src.profile

def brovey_single(pan, band):
    total = np.maximum(pan + band, 1e-6)
    return np.clip((band / total) * pan, 0, 10000)

def process_tile(safe_folder, output_base):
    if DEBUG:
        tqdm.write(f"\n[INFO] Processing tile: {safe_folder.name}")

    img_data_dirs = list(safe_folder.glob("GRANULE/*/IMG_DATA"))
    if not img_data_dirs:
        raise FileNotFoundError(f"No IMG_DATA found in {safe_folder}")
    img_dir = img_data_dirs[0]

    if DEBUG:
        tqdm.write(f"[INFO] Found IMG_DATA at: {img_dir}")

    _, profile = load_band(img_dir, "B02")
    b8, _ = load_band(img_dir, "B08")  # PAN for Brovey

    tile_name = safe_folder.stem
    tile_out = output_base / tile_name
    tile_out.mkdir(parents=True, exist_ok=True)
    if DEBUG:
        tqdm.write(f"[INFO] Output directory created: {tile_out}")

    # all bands in L1C product
    all_bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
                 "B08", "B8A", "B09", "B10", "B11", "B12"]

    # bands already in 10m
    native_10m = {"B02", "B03", "B04", "B08"}

    # bands we want to Brovey
    brovey_bands = {"B8A", "B11", "B12"}

    bar = tqdm(all_bands, desc=f"Resampling {tile_name}", leave=False, colour="blue")
    for band_id in bar:
        bar.set_postfix_str(f"{band_id}")

        path = next(img_dir.glob(f"*_{band_id}.jp2"))
        out_path = tile_out / f"{band_id}_10m.tif"

        if band_id in native_10m:
            # copy em as is
            with rasterio.open(path) as src:
                data = src.read(1)
                profile = src.profile
            profile.update(dtype=rasterio.uint16, count=1)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data.astype(np.uint16), 1)
        else:
            # resample
            with rasterio.open(path) as src:
                data = src.read(
                    out_shape=(1, b8.shape[0], b8.shape[1]),
                    resampling=Resampling.bilinear
                )[0]
                profile = src.profile
            if band_id in brovey_bands:
                data = brovey_single(b8, data)

            from rasterio.warp import calculate_default_transform

            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs, src.crs,
                src.width, src.height,
                *src.bounds,
                dst_width=b8.shape[1],
                dst_height=b8.shape[0]
            )

            profile.update(
                height=dst_height,
                width=dst_width,
                transform=dst_transform,
                dtype=rasterio.uint16,
                count=1
            )

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data.astype(np.uint16), 1)

    if DEBUG:
        tqdm.write(f"Finished tile: {tile_name}")

if __name__ == "__main__":
    cwd = Path(__file__).resolve().parent
    print(f"Current working directory: {cwd}")
    data_dir = cwd.parent / "data"
    output_dir = data_dir / "processed"
    print(f"Output directory: {output_dir}")
    output_dir.mkdir(exist_ok=True)

    safe_tiles = list(data_dir.glob("S2A_MSIL1C_*.SAFE"))
    for tile in tqdm(safe_tiles, desc="Pansharpening SAFE tiles", colour="red"):
        process_tile(tile, output_dir)

