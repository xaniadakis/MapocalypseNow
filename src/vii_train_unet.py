import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, JaccardIndex, ConfusionMatrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from v_prepare_training_data import Sentinel2Dataset, get_training_augmentations, \
    PATCH_DIR, SPLIT_DIR, label_to_cls, label_to_text
from vi_sentinel2_unet import Sentinel2UNet
from tqdm import trange
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from pathlib import Path

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

cwd = Path(__file__).resolve().parent
assets_dir = cwd.parent / "assets"

# === Setup ===
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")

BATCH_SIZE = 16
NUM_CLASSES = len(label_to_cls.values())
NUM_EPOCHS = 20
BACKBONE = "resnet50"
DROPOUT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load datasets ===
train_dataset = Sentinel2Dataset(
    split_txt=SPLIT_DIR / "train.txt",
    patch_dir=PATCH_DIR,
    transform=get_training_augmentations()
)
val_dataset = Sentinel2Dataset(
    split_txt=SPLIT_DIR / "val.txt",
    patch_dir=PATCH_DIR,
    transform=None
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# === Loss, optimizer and scheduler ===
# Get all training labels as a flat array
all_labels = np.concatenate([label.flatten() for _, label, mask in train_dataset])

# reverse lookup if all_labels uses remapped class indices
remapped = True  # set this True if your all_labels uses values like 0,1,2...
if remapped:
    remap = {v: k for k, v in label_to_cls.items()}
    label_names = {k: label_to_text.get(remap.get(k), f"Class {k}") for k in np.unique(all_labels)}
else:
    label_names = {k: label_to_text.get(k, f"Class {k}") for k in np.unique(all_labels)}
unique_vals, counts = np.unique(all_labels, return_counts=True)
total = counts.sum()
print("Label distribution:")
valid_classes_num = 0
for val, count in zip(unique_vals, counts):
    name = label_names.get(val, f"Class {val}")
    pct = 100 * count / total
    print(f"{name}: {count} pixels ({pct:.2f}%)")
    if val != 255:
        valid_classes_num += 1
print(f"We see {valid_classes_num} classes out of total {NUM_CLASSES}!")


# === Model ===
model = Sentinel2UNet(num_classes=NUM_CLASSES, backbone=BACKBONE, pretrained=True, dropout_p=DROPOUT).to(DEVICE)

# Filter valid labels
valid_labels = all_labels[all_labels != 0]
unique_classes = np.unique(valid_labels)
weights = compute_class_weight("balanced", classes=unique_classes, y=valid_labels)

all_weights = np.zeros(NUM_CLASSES, dtype=np.float32)
for cls, w in zip(unique_classes, weights):
    if cls >= NUM_CLASSES:
        raise ValueError(f"Class index {cls} out of bounds for NUM_CLASSES = {NUM_CLASSES}")
    all_weights[cls] = w

weights_tensor = torch.tensor(all_weights, dtype=torch.float).to(DEVICE)

# # === Dynamically determine used classes above a threshold ===
# all_labels = np.concatenate([label.flatten() for _, label, mask in train_dataset])
# valid_labels = all_labels[all_labels != 0]  # skip NoData
#
# # Get class distribution
# vals, counts = np.unique(valid_labels, return_counts=True)
# total_valid = counts.sum()
# class_freqs = dict(zip(vals, counts / total_valid))
#
# # Filter out classes with less than 0.5% of the data
# MIN_FREQ = 0.005
# kept_classes = sorted([cls for cls, freq in class_freqs.items() if freq > MIN_FREQ])
# print(f"Keeping {len(kept_classes)} classes (>{MIN_FREQ*100:.1f}%):", kept_classes)
#
# # Build mapping
# class_id_map = {old: new for new, old in enumerate(kept_classes)}
# NUM_CLASSES = len(kept_classes)
#
# # Update label_to_cls and label_to_text to reflect filtered classes
# filtered_label_to_cls = {k: class_id_map[v] for k, v in label_to_cls.items() if v in kept_classes}
# filtered_label_to_text = {class_id_map[v]: label_to_text[k] for k, v in label_to_cls.items() if v in kept_classes}
#
# # Remap labels in dataset
# def remap_dataset(dataset):
#     remapped = []
#     for img, lbl, msk in dataset:
#         new_lbl = np.full_like(lbl, 0)
#         for old, new in class_id_map.items():
#             new_lbl[lbl == old] = new
#         remapped.append((img, new_lbl, msk))
#     return remapped
#
# train_dataset = remap_dataset(train_dataset)
# val_dataset = remap_dataset(val_dataset)
#
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
#
# # Compute class weights
# valid_labels = np.concatenate([label.flatten() for _, label, _ in train_dataset])
# weights = compute_class_weight("balanced", classes=np.unique(valid_labels), y=valid_labels)
# weights_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)


# # Update model
# model = Sentinel2UNet(num_classes=NUM_CLASSES, backbone=BACKBONE, pretrained=True).to(DEVICE)
# criterion = nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=0)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

# === Metrics ===
acc_metric = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
iou_metric = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
cm_metric = ConfusionMatrix(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)

# === Training loop ===
train_losses, val_losses = [], []
train_ious, val_ious = [], []
train_accs, val_accs = [], []

best_val_loss = float("inf")
best_epoch = 0
patience = 3
wait = 0
stopped_at_epoch = NUM_EPOCHS

overall_pbar = trange(NUM_EPOCHS, desc="Progress", colour="red")
for epoch in overall_pbar:
    model.train()
    running_loss = 0
    acc_metric.reset()
    iou_metric.reset()

    pbar = tqdm(train_loader, desc=f"Train", colour="blue", leave=False)
    for imgs, labels, masks in pbar:
        imgs, labels, masks = imgs.to(DEVICE), labels.to(DEVICE), masks.to(DEVICE)
        labels[masks == 0] = 0
        labels = labels.long()

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            valid_mask = (labels != 0)
            if valid_mask.any():
                acc_metric.update(preds[valid_mask], labels[valid_mask])
                iou_metric.update(preds[valid_mask], labels[valid_mask])

    train_loss = running_loss / len(train_loader)
    train_acc = acc_metric.compute().item()
    train_accs.append(train_acc)
    train_iou = iou_metric.compute().item()
    train_losses.append(train_loss)
    train_ious.append(train_iou)

    # === Validation ===
    model.eval()
    val_running_loss = 0
    acc_metric.reset()
    iou_metric.reset()
    cm_metric.reset()

    with torch.no_grad():
        for imgs, labels, masks in tqdm(val_loader, desc="Eval", leave=False):
            imgs, labels, masks = imgs.to(DEVICE), labels.to(DEVICE), masks.to(DEVICE)
            labels[masks == 0] = 0
            labels = labels.long()

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            valid_mask = (labels != 0)
            if valid_mask.any():
                acc_metric.update(preds[valid_mask], labels[valid_mask])
                iou_metric.update(preds[valid_mask], labels[valid_mask])
                cm_metric.update(preds[valid_mask], labels[valid_mask])

    val_loss = val_running_loss / len(val_loader)
    val_acc = acc_metric.compute().item()
    val_accs.append(val_acc)
    val_iou = iou_metric.compute().item()
    val_losses.append(val_loss)
    val_ious.append(val_iou)
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]
    tqdm.write(
        f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f},"
        f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Val IoU={val_iou:.4f}, LR = {current_lr:.6e}"
    )

    # === Early stopping ===
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        wait = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}, because best epoch was {best_epoch}.")
            stopped_at_epoch = epoch+1
            break
    torch.cuda.empty_cache()

# === Plots ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Loss per Epoch")
plt.subplot(1, 2, 2)
plt.plot(train_ious, label="Train IoU")
plt.plot(val_ious, label="Val IoU")
plt.legend()
plt.title("IoU per Epoch")
plt.tight_layout()
plt.savefig(assets_dir / f"metrics_{BACKBONE}_{stopped_at_epoch}ep.png")

# === Confusion Matrix ===
cm = cm_metric.compute().cpu().numpy()
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Validation)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(assets_dir / f"cm_{BACKBONE}_{stopped_at_epoch}ep.png")

plt.figure(figsize=(12, 4))
plt.subplot(1, 1, 1)
plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Val Accuracy")
plt.legend()
plt.title("Accuracy per Epoch")
plt.tight_layout()
plt.savefig(assets_dir / f"accuracy_{BACKBONE}_{stopped_at_epoch}ep.png")
