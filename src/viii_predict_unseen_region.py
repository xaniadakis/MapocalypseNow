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
from vii2_train_unet import NUM_CLASSES
from vi2_sentinel2_unet import UNetResNet
from i_pansharpening import process_tile

# === Paths ===
cwd = Path(__file__).resolve().parent
data_dir = cwd.parent / "data"
safe_tile = data_dir / "S2A_MSIL1C_20210925T092031_N0500_R093_T34SEH_20230118T233535.SAFE"
processed_dir = data_dir / "processed"
tile_path = processed_dir / safe_tile.stem
ref_raw_path = data_dir / "GBDA24_ex2_34SEH_ref_data.tif"
ref_aligned_path = processed_dir / "GBDA24_ex2_34SEH_ref_data_reprojected.tif"
model_ckpt = cwd.parent / "checkpoints/20250416_163639/best_model.pth"
output_path = processed_dir / "prediction_map.tif"
patch_dir = processed_dir / "patched_test_tile"
patch_dir.mkdir(exist_ok=True)

bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
         "B08", "B8A", "B09", "B10", "B11", "B12"]

LOAD_PREDICTIONS = True

if not LOAD_PREDICTIONS:
    # === Step 1: Process SAFE tile ===
    print("📦 Processing SAFE tile...")
    process_tile(safe_tile, processed_dir)

    # === Step 2: Align and resample reference data ===
    print("🗺️  Aligning reference data...")
    with rasterio.open(tile_path / "B02_10m.tif") as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_height, dst_width = ref.height, ref.width
        dst_meta = ref.meta.copy()

    with rasterio.open(ref_raw_path) as src:
        src_array = src.read(1)
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

    # === Step 3: Save patches to disk ===
    print("🧩 Generating disk-based patches...")
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

    # Clean up RAM and VRAM
    del patch
    gc.collect()
    torch.cuda.empty_cache()

    # === Step 4: Load model ===
    print("🧠 Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetResNet(encoder_depth=101, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()

    # === Step 5: Predict from disk patches ===
    print("🔮 Running prediction from patches...")
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

    # === Step 6: Save prediction ===
    dst_meta.update({
        "count": 1,
        "dtype": "uint8",
        "driver": "GTiff",
        "compress": "lzw"
    })
    with rasterio.open(output_path, "w", **dst_meta) as dst:
        dst.write(pred_map, 1)

    print(f"✅ Prediction saved to: {output_path}")
else:
    with rasterio.open(output_path) as src:
        pred_map = src.read(1)

# === Step 7: Evaluate ===
print("📊 Evaluating prediction...")
with rasterio.open(ref_aligned_path) as ref:
    gt = ref.read(1)

valid_mask = (gt != 0)
gt_flat = gt[valid_mask]
pred_flat = pred_map[valid_mask]
gt_remapped = np.vectorize(lambda x: label_to_cls.get(x, 0))(gt_flat)

# === Build reverse mapping and filter valid classes ===
cls_to_label = {v: k for k, v in label_to_cls.items() if v != 0}
valid_model_classes = sorted(cls_to_label.keys())  # e.g. [1, 2, ..., 8]
target_names = [label_to_text[cls_to_label[i]] for i in valid_model_classes]

# === Filter only valid classes (exclude background = 0) ===
mask_valid = (gt_remapped != 0)
gt_eval = gt_remapped[mask_valid]
pred_eval = pred_flat[mask_valid]

# === Confusion Matrix ===
cm = confusion_matrix(gt_eval, pred_eval, labels=valid_model_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(xticks_rotation=45, cmap="Blues")
plt.tight_layout()
plt.show()

# === Classification Report ===
print("\n📄 Classification Report:")
print(classification_report(
    gt_eval, pred_eval,
    target_names=target_names,
    labels=valid_model_classes,
    zero_division=0
))
