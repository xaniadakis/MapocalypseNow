import torch
import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import gc

from v_prepare_training_data import label_to_cls, label_to_text
from vii_train_unet import NUM_CLASSES
from vi_sentinel2_unet import UNetResNet
from i_pansharpening import process_tile

cwd = Path(__file__).resolve().parent
data_dir = cwd.parent / "data"
safe_tile = data_dir / "S2A_MSIL1C_20210925T092031_N0500_R093_T34SEH_20230118T233535.SAFE"
processed_dir = data_dir / "processed"
tile_path = processed_dir / safe_tile.stem
ref_raw_path = data_dir / "GBDA24_ex2_34SEH_ref_data.tif"
ref_aligned_path = processed_dir / "GBDA24_ex2_34SEH_ref_data_reprojected.tif"
model_ckpt = cwd.parent / "checkpoints/20250416_163639/best_model.pth"
output_path = processed_dir / "prediction_map.tif"
output_rgb_path = processed_dir / "prediction_map_rgb.tif"
patch_dir = processed_dir / "patched_test_tile"
patch_dir.mkdir(exist_ok=True)

bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
         "B08", "B8A", "B09", "B10", "B11", "B12"]

LOAD_PREDICTIONS = True

if not LOAD_PREDICTIONS:
    # process SAFE tile
    print("Processing SAFE tile...")
    process_tile(safe_tile, processed_dir)

    # align and resample reference data
    print("Aligning reference data...")
    with rasterio.open(tile_path / "B02_10m.tif") as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_height, dst_width = ref.height, ref.width
        dst_meta = ref.meta.copy()

    with rasterio.open(ref_raw_path) as src:
        src_array = src.read(1)
        original_colormap = src.colormap(1)

        dst_array = np.zeros((dst_height, dst_width), dtype=src_array.dtype)

        reproject(
            source=src_array,
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": dst_height,
            "width": dst_width,
            "transform": dst_transform,
            "crs": dst_crs
        })

        with rasterio.open(ref_aligned_path, "w", **out_meta) as dst:
            dst.write(dst_array, 1)
            dst.write_colormap(1, original_colormap)  # <-- attach colormap

    # save patches to disk
    print("Generating disk-based patches...")
    PATCH_SIZE = 512
    STRIDE = 512

    for row in tqdm(range(0, dst_height, STRIDE), desc="Saving patches", colour="cyan"):
        for col in range(0, dst_width, STRIDE):
            patch = np.zeros((13, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            for i, b in enumerate(bands):
                with rasterio.open(tile_path / f"{b}_10m.tif") as src:
                    band_patch = src.read(1, window=Window(col, row, PATCH_SIZE, PATCH_SIZE))
                    band_patch = band_patch.astype(np.float32) / 10000.0
                    pad_h = PATCH_SIZE - band_patch.shape[0]
                    pad_w = PATCH_SIZE - band_patch.shape[1]
                    if pad_h > 0 or pad_w > 0:
                        band_patch = np.pad(band_patch, ((0, pad_h), (0, pad_w)), constant_values=0)
                    patch[i] = band_patch

            patch_path = patch_dir / f"patch_{row}_{col}.npy"
            np.save(patch_path, patch)

    # clean up RAM and VRAM
    del patch
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetResNet(encoder_depth=101, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()

    # predict from disk patches
    print("Running prediction from patches...")
    pred_map = np.zeros((dst_height, dst_width), dtype=np.uint8)
    patch_paths = sorted(patch_dir.glob("patch_*.npy"))

    for patch_path in tqdm(patch_paths, desc="Predicting", colour="green"):
        coords = patch_path.stem.split("_")[1:]
        row, col = map(int, coords)
        patch = np.load(patch_path)

        with torch.no_grad():
            tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(tensor).argmax(1).squeeze().cpu().numpy()

        h = min(PATCH_SIZE, dst_height - row)
        w = min(PATCH_SIZE, dst_width - col)
        pred_map[row:row + h, col:col + w] = pred[:h, :w]

        del tensor, pred, patch
        gc.collect()
        torch.cuda.empty_cache()

    # save pred_map to disk
    np.save("pred_map.npy", pred_map)

    # To load it later
    # pred_map = np.load("pred_map.npy")
    with rasterio.open(tile_path / "B02_10m.tif") as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_height, dst_width = ref.height, ref.width
        dst_meta = ref.meta.copy()

    # save prediction
    dst_meta.update({
        "count": 1,
        "dtype": "uint8",
        "driver": "GTiff",
        "compress": "lzw"
    })
    with rasterio.open(output_path, "w", **dst_meta) as dst:
        dst.write(pred_map, 1)
    print(f"✅ Prediction saved to: {output_path}")

    from tqdm import tqdm
    from rasterio.windows import Window
    print("🎨 Saving RGB prediction (fully optimized, windowed)...")
    with rasterio.open(output_path) as src:
        pred_profile = src.profile
        height, width = src.height, src.width
        pred_map = src.read(1)

    with rasterio.open(ref_aligned_path) as ref:
        original_colormap = ref.colormap(1)
    tile_size = 512
    meta = pred_profile.copy()
    meta.update({
        "count": 3,
        "dtype": "uint8",
        "driver": "GTiff",
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "interleave": "pixel",
        "BIGTIFF": "IF_SAFER"
    })
    if output_rgb_path.exists():
        output_rgb_path.unlink()

    cls_to_label = {v: k for k, v in label_to_cls.items() if v != 0}

    with rasterio.open(output_rgb_path, "w", **meta) as dst:
        for row in tqdm(range(0, height, tile_size), desc="Rows", colour="cyan"):
            for col in range(0, width, tile_size):
                window = Window(col, row,
                                min(tile_size, width - col),
                                min(tile_size, height - row))
                pred_tile = pred_map[row:row + window.height, col:col + window.width]
                rgb_tile = np.zeros((3, window.height, window.width), dtype='uint8')

                for cls_val, label in cls_to_label.items():
                    color = original_colormap.get(label, (0, 0, 0))
                    mask = (pred_tile == cls_val)
                    for i in range(3):
                        rgb_tile[i][mask] = color[i]

                dst.write(rgb_tile, window=window)

    print(f"RGB prediction saved to: {output_rgb_path}")
else:
    with rasterio.open(output_path) as src:
        pred_map = src.read(1)


print("Evaluating prediction...")
with rasterio.open(ref_aligned_path) as ref:
    gt = ref.read(1)

valid_mask = (gt != 0)
gt_flat = gt[valid_mask]
pred_flat = pred_map[valid_mask]
gt_remapped = np.vectorize(lambda x: label_to_cls.get(x, 0))(gt_flat)

# build reverse mapping and filter valid classes
print("Building reverse mapping...")
cls_to_label = {v: k for k, v in label_to_cls.items() if v != 0}
valid_model_classes = sorted(cls_to_label.keys())
target_names = [label_to_text[cls_to_label[i]] for i in valid_model_classes]

# filter only valid classes
print("Filtering valid classes...")
mask_valid = (gt_remapped != 0)
gt_eval = gt_remapped[mask_valid]
pred_eval = pred_flat[mask_valid]

import matplotlib.ticker as mticker

# confusion matrix with custom value formatting
print("Plotting formatted confusion matrix")
cm = confusion_matrix(gt_eval, pred_eval, labels=valid_model_classes)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=target_names
)

fig, ax = plt.subplots(figsize=(8, 6))

def format_cm_value(x):
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    elif x >= 1_000:
        return f"{x / 1_000:.1f}K"
    else:
        return str(x)

disp.plot(
    include_values=False,
    cmap="Blues",
    xticks_rotation=45,
    ax=ax
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format_cm_value(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=8)

plt.tight_layout()
plt.show()


print("\nClassification Report:")
print(classification_report(
    gt_eval, pred_eval,
    target_names=target_names,
    labels=valid_model_classes,
    zero_division=0
))