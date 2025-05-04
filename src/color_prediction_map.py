from pathlib import Path
import rasterio

cwd = Path(__file__).resolve().parent
data_dir = cwd.parent / "data"
processed_dir = data_dir / "processed"
output_path = processed_dir / "prediction_map.tif"
grayscale_path = output_path
ref_path = processed_dir / "GBDA24_ex2_34SEH_ref_data_reprojected.tif"
colored_output_path = processed_dir / "prediction_map_colored.tif"


colormap = None

# Read the colormap from reference raster
with rasterio.open(ref_path) as ref:
    colormap = ref.colormap(1)
    meta = ref.meta.copy()

    print("Colormap keys:", list(colormap.keys()))

# Apply colormap to prediction
with rasterio.open(grayscale_path) as src:
    pred_data = src.read(1)
    pred_meta = src.meta.copy()

    # Update metadata for colormap
    pred_meta.update({
        'count': 1,
        'dtype': 'uint8',
        'photometric': 'palette'
    })

    with rasterio.open(colored_output_path, 'w', **pred_meta) as dst:
        dst.write(pred_data, 1)
        dst.write_colormap(1, colormap)

print(f"✅ Colorized prediction saved at: {colored_output_path}")
