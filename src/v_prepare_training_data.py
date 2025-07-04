import random
from pathlib import Path
import matplotlib.patches as mpatches
from rasterio.windows import Window
import cv2
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import albumentations as A

PATCH_SIZE = STRIDE = 512
PATCH_DIR = Path(f"data/patch_dataset_{PATCH_SIZE}_{STRIDE}")
AUG_PATCH_DIR = Path(f"data/augmented_dataset_{PATCH_SIZE}_{STRIDE}")
AUG_PATCH_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_DIR = PATCH_DIR / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
cwd = Path(__file__).resolve().parent
assets_dir = cwd.parent / "assets"
data_dir = cwd.parent / "data"
processed_dir = data_dir / "processed"
ref_path = processed_dir / "GBDA24_ex2_ref_data_reprojected.tif"

dataset_dir = data_dir / f"patch_dataset_{PATCH_SIZE}_{STRIDE}"
mask_dir = dataset_dir / "masks"
mask_paths = sorted(mask_dir.glob("mask_*.tif"))

SPLIT_RATIOS = (0.7, 0.2, 0.1)
SEED = 42

label_to_cls = {
    0: 0,     # NoData / Background (if needed)
    10: 1,    # Tree cover
    20: 2,    # Shrubland
    30: 3,    # Grassland
    40: 4,    # Cropland
    50: 5,    # Built-up
    60: 6,    # Bare/sparse vegetation
    # 70: 7,    # Snow and ice
    80: 7,    # Permanent water bodies
    90: 8,    # Herbaceous wetland
    # 95: 10,   # Mangroves
    # 100: 11   # Moss and lichen
}
label_to_text = {
    0: "No Data",
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare land",
    # 70: "Snow/Ice",
    80: "Water",
    90: "Wetland",
    # 95: "Mangroves",
    # 100: "Moss/Lichen"
}


def create_split_files():
    image_paths = sorted((PATCH_DIR / "images").glob("image_*.tif"))
    ids = [p.stem.replace("image_", "") for p in image_paths]
    random.seed(SEED)
    random.shuffle(ids)

    n_total = len(ids)
    n_train = int(n_total * SPLIT_RATIOS[0])
    n_val = int(n_total * SPLIT_RATIOS[1])

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    for split, name in zip([train_ids, val_ids, test_ids], ["train", "val", "test"]):
        with open(SPLIT_DIR / f"{name}.txt", "w") as f:
            f.writelines(f"{i}\n" for i in split)

    print(f"Split complete: {len(train_ids)} train | {len(val_ids)} val | {len(test_ids)} test.")
    print(f"Total dataset size: {len(train_ids)+len(val_ids)+len(test_ids)} patches.")

class Sentinel2Dataset(Dataset):
    def __init__(self, split_txt, patch_dir, transform=None,
                 downsample_size=None):
        self.patch_dir = Path(patch_dir)
        with open(split_txt) as f:
            self.ids = [line.strip() for line in f]
        self.transform = transform
        self.downsample_size = downsample_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        img_path = self.patch_dir / "images" / f"image_{id_}.tif"
        lbl_path = self.patch_dir / "labels" / f"label_{id_}.tif"
        msk_path = self.patch_dir / "masks" / f"mask_{id_}.tif"


        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32) / 10000.0

        with rasterio.open(lbl_path) as src:
            label = src.read(1)

        label = np.vectorize(lambda x: label_to_cls.get(x, 0))(label)

        with rasterio.open(msk_path) as src:
            mask = src.read(1).astype(np.uint8)

        # Add downsampling before transformations
        if self.downsample_size and image.shape[1] > self.downsample_size:
            # Resize the image and label
            image_resized = np.zeros((image.shape[0], self.downsample_size, self.downsample_size), dtype=np.float32)
            for i in range(image.shape[0]):
                image_resized[i] = cv2.resize(image[i], (self.downsample_size, self.downsample_size))

            label = cv2.resize(label, (self.downsample_size, self.downsample_size),
                               interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (self.downsample_size, self.downsample_size),
                              interpolation=cv2.INTER_NEAREST)

            image = image_resized

        if self.transform:
            augmented = self.transform(image=image.transpose(1, 2, 0), mask=label)
            image = augmented["image"].transpose(2, 0, 1)
            label = augmented["mask"]

        return image, label, mask

def get_training_augmentations(seed=42, means=None, stds=None):
    if means is None:
        means = np.array([
            0.2153, 0.1946, 0.1852, 0.1804, 0.2002, 0.2572, 0.2813,
            0.2819, 0.1583, 0.1939, 0.1173, 0.1491, 0.1327
        ])
    if stds is None:
        stds = np.array([
            0.1320, 0.1275, 0.1261, 0.1290, 0.1350, 0.1563, 0.1683,
            0.1700, 0.1225, 0.1310, 0.1030, 0.1197, 0.1130
        ])
    return A.Compose([
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(0.1, 0.1),
            rotate=(-15, 15),
            shear=(-10, 10),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.4
        ),
        # Radiometric augmentations
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.GaussNoise(std_range=(0.03, 0.1), mean_range=(0.0, 0.0), per_channel=True, p=0.3),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, elementwise=True, p=0.3),
        # normalization
        A.Normalize(mean=means.tolist(), std=stds.tolist(), max_pixel_value=1.0)
    ], seed=seed)

from collections import Counter

def plot_all_split_frequencies():
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 12),
        gridspec_kw={"height_ratios": [1, 1, 1], "hspace": 0.3}
    )
    split_files = [("Train", SPLIT_DIR / "train.txt"),
                   ("Val", SPLIT_DIR / "val.txt"),
                   ("Test", SPLIT_DIR / "test.txt")]

    for ax, (name, split_txt) in zip(axes, split_files):
        ds = Sentinel2Dataset(split_txt=split_txt, patch_dir=PATCH_DIR, transform=None)
        label_counter = Counter()

        for _, label, mask in ds:
            valid = label[(mask == 1) & (label != 0)]
            unique, counts = np.unique(valid, return_counts=True)
            label_counter.update(dict(zip(unique, counts)))

        labels = sorted(label_counter.keys())
        counts = [label_counter[l] for l in labels]
        total = sum(counts)
        percentages = [100 * c / total for c in counts]

        reverse_label_mapping = {v: k for k, v in label_to_cls.items()}
        label_names = [label_to_text.get(reverse_label_mapping.get(lbl), f"Class {lbl}") for lbl in labels]

        present_classes = [label_to_text.get(reverse_label_mapping.get(lbl), f"Class {lbl}") for lbl in labels]
        print(f"{name} set has {len(present_classes)} unique classes present:")
        for cls_name in present_classes:
            print(f"  - {cls_name}")

        bars = ax.bar(label_names, counts, width=0.4)
        ax.set_title(f"{name}", fontsize=10)
        ax.set_ylabel("Pixels", fontsize=8)
        ax.set_xticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=12, ha="right", fontsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ymax = max(counts) * 1.1
        ax.set_ylim(top=ymax)

        for bar, pct in zip(bars, percentages):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total * 0.003,
                    f"{pct:.2f}%",
                    ha='center', va='bottom', fontsize=6)

    plt.tight_layout()
    plt.savefig(assets_dir / "dataset_splits_class_freqs.png")

def plot_grid():
    print("\nBuilding downsampled mask grid...")
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
    target_size = 64
    masks = []
    for path in mask_paths:
        with rasterio.open(path) as src:
            m = src.read(1)
        m_ds = cv2.resize(m, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        masks.append(m_ds)

    print(f"Rows: {row_count}, Cols: {col_count}")

    canvas_rgb = np.zeros((row_count * target_size, col_count * target_size, 3), dtype=np.uint8)

    train_ids = set(open(SPLIT_DIR / "train.txt").read().splitlines())
    val_ids = set(open(SPLIT_DIR / "val.txt").read().splitlines())
    test_ids = set(open(SPLIT_DIR / "test.txt").read().splitlines())

    color_map = {
        "train": [66, 133, 244],  # blue
        "val": [255, 165, 0],  # orange
        "test": [52, 168, 83]  # green
    }

    for path_idx, (path, mask) in enumerate(zip(mask_paths, masks)):
        patch_id = path.stem.replace("mask_", "")
        idx = int(patch_id)
        r = idx // col_count
        c = idx % col_count

        if patch_id in train_ids:
            color = color_map["train"]
        elif patch_id in val_ids:
            color = color_map["val"]
        elif patch_id in test_ids:
            color = color_map["test"]
        else:
            continue  # not in any split

        color_mask = np.stack([(mask == 1) * ch for ch in color], axis=-1).astype(np.uint8)
        canvas_rgb[r * target_size:(r + 1) * target_size,
        c * target_size:(c + 1) * target_size] = color_mask

    plt.figure(figsize=(10, 10))
    plt.imshow(canvas_rgb)
    plt.title("Split Overlay on Downsampled Mask Grid")
    plt.axis("off")

    legend_patches = [
        mpatches.Patch(color=np.array(color_map["train"]) / 255, label='Train'),
        mpatches.Patch(color=np.array(color_map["val"]) / 255, label='Val'),
        mpatches.Patch(color=np.array(color_map["test"]) / 255, label='Test')
    ]
    plt.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    create_split_files()

    print(f"Our dataset dir is {PATCH_DIR}")

    train_dataset = Sentinel2Dataset(
        split_txt=SPLIT_DIR / "train.txt",
        patch_dir=PATCH_DIR,
        transform=get_training_augmentations()
    )

    plot_grid()

    plot_all_split_frequencies()
