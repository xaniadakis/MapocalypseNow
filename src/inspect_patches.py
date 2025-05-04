from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from iv_build_images import PATCH_SIZE, STRIDE
from v_prepare_training_data import Sentinel2Dataset, PATCH_DIR, SPLIT_DIR

LANGUAGE = "en"  # change to "en" or "gr"

label_texts = {
    "en": {
        1: "Tree cover",
        2: "Shrubland",
        3: "Grassland",
        4: "Cropland",
        5: "Built-up",
        6: "Bare/Sparse vegetation",
        7: "Snow and Ice",
        8: "Permanent water bodies",
        9: "Herbaceous wetland",
        # 95: "Mangroves",
        # 100: "Moss and Lichen"
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

cwd = Path(__file__).resolve().parent
data_dir = cwd.parent / "data"
processed_dir = data_dir / "processed"
dataset_dir = data_dir / f"patch_dataset_{PATCH_SIZE}_{STRIDE}"
mask_dir = dataset_dir / "masks"
ref_path = processed_dir / "GBDA24_ex2_ref_data_reprojected.tif"
mask_paths = sorted(mask_dir.glob("mask_*.tif"))

dataset = Sentinel2Dataset(
    split_txt=SPLIT_DIR / "train.txt",
    patch_dir=PATCH_DIR,
    transform=None
)


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


import matplotlib.pyplot as plt
import numpy as np
import random

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

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle(f"Random Patch #{i}", fontsize=10, fontweight='bold')

    plot_data = [
        (rgb, "RGB", None, {}),
        (label, "Label", "tab20", {}),
        (mask, "Mask", "gray", {"vmin": 0, "vmax": 1})
    ]

    for ax, (data, title, cmap, kwargs) in zip(axes.flat[:3], plot_data):
        ax.imshow(data, cmap=cmap, **kwargs)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("gray")
            spine.set_linewidth(0.8)

    ax_hist = axes[1][1]
    ax_hist.set_box_aspect(1)
    percentages = 100 * counts / total
    class_names = [label_names.get(l, f"Class {l}") for l in unique_labels]
    bars = ax_hist.bar(class_names, percentages, color='steelblue', edgecolor='black', linewidth=0.6)

    ax_hist.set_title("Label Distribution (%)", fontsize=12)
    ax_hist.grid(axis="y", linestyle="--", alpha=0.4)
    plt.setp(ax_hist.get_xticklabels(), rotation=20, ha="right", fontsize=8)

    for bar, pct in zip(bars, percentages):
        ax_hist.annotate(
            f"{pct:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 0.5),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=8
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    plt.show()
