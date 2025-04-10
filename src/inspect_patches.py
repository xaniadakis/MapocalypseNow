import random
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import cv2
import rasterio
from rasterio.windows import Window
from iv_build_images import PATCH_SIZE
from v_prepare_training_data import Sentinel2Dataset, PATCH_DIR, SPLIT_DIR

# === CONFIG ===
LANGUAGE = "gr"  # change to "en" or "gr"

label_texts = {
    "en": {
        1: "Tree cover",
        2: "Shrubland",
        3: "Grassland",
        4: "Cropland",
        5: "Built-up",
        6: "Bare land",
        7: "Snow/Ice",
        8: "Water",
        9: "Wetland",
    },
    "gr": {
        1: "Δάσος",
        2: "Θάμνοι",
        3: "Λιβάδια",
        4: "Καλλιέργειες",
        5: "Δόμηση",
        6: "Γυμνή γη",
        7: "Χιόνι/Πάγος",
        8: "Νερό",
        9: "Yγρότοποι",
    }
}
label_names = label_texts[LANGUAGE]

# === PATHS ===
cwd = Path(__file__).resolve().parent
data_dir = cwd.parent / "data"
processed_dir = data_dir / "processed"
dataset_dir = data_dir / "patch_dataset"
mask_dir = dataset_dir / "masks"
ref_path = processed_dir / "GBDA24_ex2_ref_data_reprojected.tif"
mask_paths = sorted(mask_dir.glob("mask_*.tif"))

# === Load Dataset ===
dataset = Sentinel2Dataset(
    split_txt=SPLIT_DIR / "train.txt",
    patch_dir=PATCH_DIR,
    transform=None
)


# === Count empty/full/mixed masks ===
empty = 0
full = 0
mixed = 0
print("Checking masks...\n")
for path in mask_paths:
    with rasterio.open(path) as src:
        mask = src.read(1)
    unique_vals = np.unique(mask)
    if np.array_equal(unique_vals, [0]):
        empty += 1
    elif np.array_equal(unique_vals, [1]):
        full += 1
    elif set(unique_vals).issubset({0, 1}):
        mixed += 1
    else:
        print(f"{path.name} contains unexpected values: {unique_vals}")

print(f"Total masks: {len(mask_paths)}")
print(f"Empty (all 0): {empty}")
print(f"Full  (all 1): {full}")
print(f"Mixed (0 + 1): {mixed}")




# === Create low-res mask grid plot ===
print("\nBuilding downsampled mask grid...")
# Plot whole grid
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
target_size = 16
masks = []
for path in mask_paths:
    with rasterio.open(path) as src:
        m = src.read(1)
    m_ds = cv2.resize(m, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    masks.append(m_ds)
print(f"Rows: {row_count}, Cols: {col_count}")
canvas = np.zeros((row_count * target_size, col_count * target_size), dtype=np.uint8)
for idx, mask in enumerate(masks):
    r = idx // col_count
    c = idx % col_count
    canvas[r * target_size : (r + 1) * target_size, c * target_size : (c + 1) * target_size] = mask
# Show low-res mask grid
plt.figure(figsize=(10, 10))
plt.imshow(canvas, cmap="gray")
plt.title("Downsampled Valid Mask Grid")
plt.axis("off")
plt.show()



# === Label Frequency Across Dataset ===
label_counter = Counter()
for _, label, mask in dataset:
    valid_label = label[(mask == 1) & (label != 255)]
    unique, counts = np.unique(valid_label, return_counts=True)
    label_counter.update(dict(zip(unique, counts)))

total_valid_pixels = sum(label_counter.values())

print("\nTotal label frequencies across dataset (255 ignored):")
for lbl, count in sorted(label_counter.items()):
    pct = 100 * count / total_valid_pixels
    label_str = label_names.get(lbl, f"Class {lbl}")
    print(f"{label_str}: {count} pixels ({pct:.2f}%)")

# Plot global distribution
plt.figure(figsize=(10, 5))
labels = sorted([lbl for lbl in label_counter if lbl != 255])
frequencies = [label_counter[lbl] for lbl in labels]
percentages = [100 * count / total_valid_pixels for count in frequencies]
text_labels = [label_names.get(lbl, f"Class {lbl}") for lbl in labels]

bars = plt.bar(text_labels, frequencies)
plt.title("Label Frequency Across All Patches")
plt.xlabel("Label")
plt.ylabel("Pixel Count")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks(rotation=15, ha="right")

for bar, pct in zip(bars, percentages):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2,
             height + total_valid_pixels * 0.005,
             f"{pct:.1f}%",
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# === Random Patch Visualizations ===
num_to_show = 30
indices = random.sample(range(len(dataset)), num_to_show)

for i in indices:
    img, label, mask = dataset[i]
    rgb = img[[3, 2, 1], :, :]
    rgb = rgb / rgb.max()
    rgb = rgb.transpose(1, 2, 0)

    valid_label = label[(mask == 1) & (label != 255)]
    unique_labels, counts = np.unique(valid_label, return_counts=True)
    total = counts.sum()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    titles = ["RGB", "Label", "Mask", "Label Distribution (%)"]
    images = [rgb, label, mask]
    cmaps = [None, "tab20", "gray"]
    extras = [{}, {}, {"vmin": 0, "vmax": 1}]

    for ax, img_data, title, cmap, extra in zip(axes[:3], images, titles[:3], cmaps, extras):
        ax.imshow(img_data, cmap=cmap, **extra)
        ax.set_title(title)
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1)

    # Histogram for patch
    labels_nonzero = unique_labels
    counts_nonzero = counts
    percentages = 100 * counts_nonzero / total
    text_labels = [label_names.get(lbl, f"Class {lbl}") for lbl in labels_nonzero]

    bars = axes[3].bar(text_labels, percentages)
    axes[3].set_title(titles[3])
    axes[3].set_xlabel("Label", fontsize=8)
    axes[3].set_ylabel("Percentage (%)", fontsize=8)
    axes[3].grid(axis="y", linestyle="--", alpha=0.5)
    plt.setp(axes[3].get_xticklabels(), rotation=15, ha="right", fontsize=7)

    for bar, pct in zip(bars, percentages):
        axes[3].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1,
                     f"{pct:.1f}%",
                     ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"Random Patch #{i}")
    plt.tight_layout()
    plt.show()

# from rasterio.windows import Window
# from iv_build_images import PATCH_SIZE
# from v_prepare_training_data import Sentinel2Dataset, PATCH_DIR, SPLIT_DIR
# import numpy as np
# import rasterio
# from pathlib import Path
# from math import ceil
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from torchmetrics import JaccardIndex
#
# # Load dataset (if needed for additional checks)
# dataset = Sentinel2Dataset(
#     split_txt=SPLIT_DIR / "train.txt",
#     patch_dir=PATCH_DIR,
#     transform=None
# )
#
# cwd = Path(__file__).resolve().parent
# data_dir = cwd.parent / "data"
# processed_dir = data_dir / "processed"
# dataset_dir = data_dir / "patch_dataset"
# image_dir = dataset_dir / "images"
# label_dir = dataset_dir / "labels"
# mask_dir = dataset_dir / "masks"
# ref_path = processed_dir / "GBDA24_ex2_ref_data_reprojected.tif"
#
# mask_paths = sorted(mask_dir.glob("mask_*.tif"))
#
# # === Count empty/full/mixed masks ===
# empty = 0
# full = 0
# mixed = 0
#
# print("Checking masks...\n")
# for path in mask_paths:
#     with rasterio.open(path) as src:
#         mask = src.read(1)
#     unique_vals = np.unique(mask)
#     if np.array_equal(unique_vals, [0]):
#         empty += 1
#     elif np.array_equal(unique_vals, [1]):
#         full += 1
#     elif set(unique_vals).issubset({0, 1}):
#         mixed += 1
#     else:
#         print(f"{path.name} contains unexpected values: {unique_vals}")
#
# print(f"Total masks: {len(mask_paths)}")
# print(f"Empty (all 0): {empty}")
# print(f"Full  (all 1): {full}")
# print(f"Mixed (0 + 1): {mixed}")
#
# # === Create low-res mask grid plot ===
# print("\nBuilding downsampled mask grid...")
#
# # Plot whole grid
# with rasterio.open(ref_path) as src:
#     data = src.read(1)
#     nodata = src.nodata
#     height, width = src.height, src.width
# valid_mask = data != nodata
# patch_count = 0
# row_count = 0
# col_count = 0
# for row_idx, row in enumerate(range(0, height, PATCH_SIZE)):
#     col_valid = 0
#     for col_idx, col in enumerate(range(0, width, PATCH_SIZE)):
#         window = Window(col, row,
#                         min(PATCH_SIZE, width - col),
#                         min(PATCH_SIZE, height - row))
#         patch = valid_mask[
#             int(window.row_off):int(window.row_off + window.height),
#             int(window.col_off):int(window.col_off + window.width)
#         ]
#         if patch.any():
#             patch_count += 1
#             col_valid += 1
#     if col_valid > 0:
#         row_count += 1
#         col_count = max(col_count, col_valid)
# print(f"Total valid patches: {patch_count}")
# print(f"Grid size: {row_count} rows × {col_count} cols")
#
# target_size = 16
# masks = []
# for path in mask_paths:
#     with rasterio.open(path) as src:
#         m = src.read(1)
#     m_ds = cv2.resize(m, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
#     masks.append(m_ds)
#
# print(f"Rows: {row_count}, Cols: {col_count}")
# canvas = np.zeros((row_count * target_size, col_count * target_size), dtype=np.uint8)
# for idx, mask in enumerate(masks):
#     r = idx // col_count
#     c = idx % col_count
#     canvas[r * target_size : (r + 1) * target_size, c * target_size : (c + 1) * target_size] = mask
# # Show low-res mask grid
# plt.figure(figsize=(10, 10))
# plt.imshow(canvas, cmap="gray")
# plt.title("Downsampled Valid Mask Grid")
# plt.axis("off")
# plt.show()
#
#
# from collections import Counter
# import numpy as np
# import matplotlib.pyplot as plt
#
# label_counter = Counter()
# per_patch_label_counts = []
#
# # Count all valid (non-255) labels
# for _, label, mask in dataset:
#     valid_label = label[(mask == 1) & (label != 255)]
#     unique, counts = np.unique(valid_label, return_counts=True)
#     label_counter.update(dict(zip(unique, counts)))
#     per_patch_label_counts.append(dict(zip(unique, counts)))
#
# # Compute total valid pixels
# total_valid_pixels = sum(label_counter.values())
#
# print("\nTotal label frequencies across dataset (255 ignored):")
# for lbl, count in sorted(label_counter.items()):
#     pct = 100 * count / total_valid_pixels
#     print(f"Label {lbl}: {count} pixels ({pct:.2f}%)")
#
# plt.figure(figsize=(10, 5))
# labels = sorted([lbl for lbl in label_counter if lbl != 255])
# frequencies = [label_counter[lbl] for lbl in labels]
# percentages = [100 * count / total_valid_pixels for count in frequencies]
#
# bars = plt.bar(labels, frequencies, tick_label=labels)
# plt.title("Label Frequency Across All Patches")
# plt.xlabel("Class Label")
# plt.ylabel("Pixel Count")
# plt.grid(axis='y', linestyle='--', alpha=0.5)
#
# # Use a small constant offset (in pixels) from top of each bar
# for bar, pct in zip(bars, percentages):
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width() / 2,
#              height + total_valid_pixels * 0.005,  # add 1% of total for spacing
#              f"{pct:.1f}%",
#              ha='center', va='bottom', fontsize=10)
#
# plt.tight_layout()
# plt.show()
#
# import random
# import numpy as np
# import matplotlib.pyplot as plt
#
# # ESA WorldCover class label mapping
# esa_label_names = {
#     1: "Tree cover",
#     2: "Shrubland",
#     3: "Grassland",
#     4: "Cropland",
#     5: "Built-up",
#     6: "Bare land",
# }
#
# num_to_show = 30
# indices = random.sample(range(len(dataset)), num_to_show)
#
# for i in indices:
#     img, label, mask = dataset[i]
#     rgb = img[[3, 2, 1], :, :]
#     rgb = rgb / rgb.max()
#     rgb = rgb.transpose(1, 2, 0)
#
#     valid_label = label[(mask == 1) & (label != 255)]
#     unique_labels, counts = np.unique(valid_label, return_counts=True)
#     total = counts.sum()
#
#     fig, axes = plt.subplots(1, 4, figsize=(18, 4))
#     titles = ["RGB", "Label", "Mask", "Label Distribution (%)"]
#     images = [rgb, label, mask]
#     cmaps = [None, "tab20", "gray"]
#     extras = [{}, {}, {"vmin": 0, "vmax": 1}]
#
#     for ax, img_data, title, cmap, extra in zip(axes[:3], images, titles[:3], cmaps, extras):
#         ax.imshow(img_data, cmap=cmap, **extra)
#         ax.set_title(title)
#         ax.axis("on")
#         ax.set_xticks([])
#         ax.set_yticks([])
#         for spine in ax.spines.values():
#             spine.set_visible(True)
#             spine.set_color("black")
#             spine.set_linewidth(1)
#
#     # Plot percentage histogram with Greek labels
#     nonzero = counts > 0
#     labels_nonzero = unique_labels[nonzero]
#     counts_nonzero = counts[nonzero]
#     percentages = 100 * counts_nonzero / total
#     text_labels = [esa_label_names.get(lbl, f"Class {lbl}") for lbl in labels_nonzero]
#
#     bars = axes[3].bar(text_labels, percentages)
#     axes[3].set_title(titles[3])
#     axes[3].set_xlabel("Label")
#     axes[3].set_ylabel("Percentage (%)")
#     axes[3].grid(axis="y", linestyle="--", alpha=0.5)
#     plt.setp(axes[3].get_xticklabels(), rotation=30, ha="right")
#
#     # Annotate bars with percentages
#     for bar, pct in zip(bars, percentages):
#         axes[3].text(bar.get_x() + bar.get_width() / 2,
#                      bar.get_height() + 1,
#                      f"{pct:.1f}%",
#                      ha="center", va="bottom", fontsize=9)
#
#     fig.suptitle(f"Random Patch #{i}")
#     plt.tight_layout()
#     plt.show()
#
#
#
#
# # for i in range(len(dataset)):
# #     img, label, mask = dataset[i]
# #
# #     rgb = img[[3, 2, 1], :, :]  # B04, B03, B02
# #     rgb = rgb / rgb.max()
# #     rgb = rgb.transpose(1, 2, 0)
# #
# #     fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# #     titles = ["RGB", "Label", "Mask"]
# #     images = [rgb, label, mask]
# #     cmaps = [None, "tab20", "gray"]
# #     extras = [{}, {}, {"vmin": 0, "vmax": 1}]  # for correct mask display
# #
# #     for ax, img_data, title, cmap, extra in zip(axes, images, titles, cmaps, extras):
# #         ax.imshow(img_data, cmap=cmap, **extra)
# #         ax.set_title(title)
# #         ax.axis("on")
# #         ax.set_xticks([])
# #         ax.set_yticks([])
# #         ax.spines["top"].set_visible(True)
# #         ax.spines["bottom"].set_visible(True)
# #         ax.spines["left"].set_visible(True)
# #         ax.spines["right"].set_visible(True)
# #         ax.spines["top"].set_color("black")
# #         ax.spines["bottom"].set_color("black")
# #         ax.spines["left"].set_color("black")
# #         ax.spines["right"].set_color("black")
# #         ax.spines["top"].set_linewidth(1)
# #         ax.spines["bottom"].set_linewidth(1)
# #         ax.spines["left"].set_linewidth(1)
# #         ax.spines["right"].set_linewidth(1)
# #
# #     plt.suptitle(f"Patch #{i}")
# #     plt.tight_layout()
# #     plt.show()