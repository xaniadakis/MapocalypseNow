from pathlib import Path
import matplotlib.pyplot as plt

import rasterio
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, jaccard_score
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random
import tkinter as tk
from tkinter import filedialog
import argparse
from tqdm import tqdm
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex, MulticlassAccuracy
from v_prepare_training_data import Sentinel2Dataset, get_training_augmentations, PATCH_DIR, SPLIT_DIR, label_to_cls, \
    label_to_text
from vi_sentinel2_unet import UNetResNet, DiceLoss, AdaptiveLoss
import os
from torch.utils.data import Subset
from itertools import cycle

NUM_CLASSES = 9
PATCH_SIZE = STRIDE = 512
PATCH_DIR = Path(f"data/patch_dataset_{PATCH_SIZE}_{STRIDE}")
RANDOM_SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def color(text, style='bold', color='cyan'):
    styles = {'bold': '1', 'dim': '2', 'normal': '22'}
    colors = {'cyan': '36', 'magenta': '35', 'yellow': '33', 'blue': '34'}
    return f"\033[{styles[style]};{colors[color]}m{text}\033[0m"

def get_pixel_class_distribution(dataset, patch_dir, num_classes, label_to_cls):
    total_counts = np.zeros(num_classes)
    for id_ in dataset.ids:
        lbl_path = patch_dir / "labels" / f"label_{id_}.tif"
        with rasterio.open(lbl_path) as src:
            lbl = src.read(1)
        unique, counts = np.unique(lbl, return_counts=True)
        for raw_label, count in zip(unique, counts):
            if raw_label in label_to_cls:
                cls_idx = label_to_cls[raw_label]
                total_counts[cls_idx] += count
    return total_counts

def compute_pixel_based_patch_weights(dataset, patch_dir, label_to_cls, num_classes):
    class_pixel_counts = np.zeros(num_classes)
    patch_weights = []

    for id_ in dataset.ids:
        lbl_path = patch_dir / "labels" / f"label_{id_}.tif"
        with rasterio.open(lbl_path) as src:
            lbl = src.read(1)

        patch_class_counts = np.zeros(num_classes)
        for raw_label in np.unique(lbl):
            if raw_label in label_to_cls:
                cls_idx = label_to_cls[raw_label]
                count = np.sum(lbl == raw_label)
                patch_class_counts[cls_idx] += count

        class_pixel_counts += patch_class_counts
        patch_weights.append(patch_class_counts)

    class_freq = class_pixel_counts / class_pixel_counts.sum()
    class_weights = 1.0 / (class_freq + 1e-6)

    weights = []
    for patch_dist in patch_weights:
        patch_weight = np.dot(patch_dist, class_weights)
        weights.append(patch_weight)

    return weights

def train_model(model, train_dataset, val_dataset, device, hyperparams, output_dir, early_patience=3):
    model_path = os.path.join(output_dir, "best_model.pth")
    log_file = os.path.join(output_dir, "training_log.txt")
    meta_file = os.path.join(output_dir, "best_model_meta.txt")
    plot_path = os.path.join(output_dir, "training_metrics.png")
    epochs_no_improve = 0

    torch.cuda.empty_cache()

    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparams['batch_size'],
        shuffle=True,
        num_workers=4,
        worker_init_fn=lambda _: np.random.seed(RANDOM_SEED),
        generator=g
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparams['batch_size'],
        shuffle=False,
        num_workers=4
    )

    cls_idx_to_label = {v: k for k, v in label_to_cls.items()}
    dist = get_pixel_class_distribution(train_dataset, PATCH_DIR, NUM_CLASSES, label_to_cls)
    total_pixels = dist.sum()

    def format_count(n):
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        elif n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)

    stats = []
    for i, count in enumerate(dist.astype(int)):
        raw_label = cls_idx_to_label.get(i, i)
        cls_name = label_to_text.get(raw_label, f"Class {raw_label}")
        percentage = 100 * count / total_pixels
        stats.append((percentage, i, cls_name, count))

    stats.sort(reverse=True)

    print("\nClass Distribution:")
    for percentage, i, cls_name, count in stats:
        print(f"{cls_name:15} (Cl.{i}): {format_count(count):>7}  ({percentage:5.2f}%)")
    print("")


    f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average='weighted', ignore_index=0).to(device)
    iou_metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average='weighted', ignore_index=0).to(device)
    acc_metric = MulticlassAccuracy(num_classes=NUM_CLASSES, average='micro', ignore_index=0).to(device)

    weights = 1.0 / (dist + 1e-6)  # Avoid div by zero
    weights /= weights.sum()  # Normalize to sum to 1
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    adaptive_loss = AdaptiveLoss(class_weights=class_weights.to(device), num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    # Histories
    best_f1 = 0.0
    train_loss_history = []
    val_loss_history = []
    train_f1_history = []
    val_f1_history = []
    train_acc_history = []
    val_acc_history = []
    iou_history = []

    for epoch in range(hyperparams['epochs']):
        # # Refill & reshuffle when all indices have been used
        # if index_pointer + max_patches_per_epoch > len(all_indices):
        #     random.shuffle(all_indices)
        #     index_pointer = 0
        # subset_indices = all_indices[index_pointer:index_pointer + max_patches_per_epoch]
        # index_pointer += max_patches_per_epoch
        # subset_train_dataset = Subset(train_dataset, subset_indices)
        # train_loader = DataLoader(
        #     subset_train_dataset,
        #     batch_size=hyperparams['batch_size'],
        #     shuffle=True,
        #     num_workers=4
        # )

        model.train()
        epoch_train_loss = 0.0
        train_preds = []
        train_labels = []

        # for images, labels, masks in train_loader:
        for images, labels, masks in tqdm(train_loader, desc=f"Train Epoch {epoch + 1}", leave=False, colour="blue"):
            images = images.to(device)
            labels = labels.to(device).long()
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            # loss = criterion(outputs, labels)
            loss = adaptive_loss(outputs, labels, masks)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            valid_mask = masks.bool()
            train_preds.extend(preds[valid_mask].cpu().numpy())
            train_labels.extend(labels[valid_mask].cpu().numpy())
            del images, labels, masks, outputs, preds, valid_mask, loss

        train_loss = epoch_train_loss / len(train_loader.dataset)
        # del train_loader
        torch.cuda.empty_cache()

        model.eval()
        epoch_val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            # for images, labels, masks in val_loader:
            for images, labels, masks in tqdm(val_loader, desc=f"Val Epoch {epoch + 1}", leave=False, colour="red"):
                images = images.to(device)
                labels = labels.to(device).long()
                masks = masks.to(device)

                outputs = model(images)
                # loss = criterion(outputs, labels)
                loss = adaptive_loss(outputs, labels, masks)
                epoch_val_loss += loss.item() * images.size(0)

                preds = torch.argmax(outputs, dim=1)
                valid_mask = masks.bool()
                val_preds.extend(preds[valid_mask].cpu().numpy())
                val_labels.extend(labels[valid_mask].cpu().numpy())
                del images, labels, masks, outputs, preds, valid_mask, loss

        val_loss = epoch_val_loss / len(val_loader.dataset)
        # del val_loader
        torch.cuda.empty_cache()

        with tqdm(total=6, desc="Evaluating", colour="yellow", leave=False) as pbar:
            # Convert lists to tensors on GPU
            train_preds_tensor = torch.tensor(train_preds, device=device)
            train_labels_tensor = torch.tensor(train_labels, device=device)
            val_preds_tensor = torch.tensor(val_preds, device=device)
            val_labels_tensor = torch.tensor(val_labels, device=device)
            pbar.update(1);pbar.refresh()

            train_f1 = f1_metric(train_preds_tensor, train_labels_tensor).item()
            pbar.update(1);pbar.refresh()

            val_f1 = f1_metric(val_preds_tensor, val_labels_tensor).item()
            pbar.update(1);pbar.refresh()

            iou = iou_metric(val_preds_tensor, val_labels_tensor).item()
            pbar.update(1);pbar.refresh()

            train_acc = acc_metric(train_preds_tensor, train_labels_tensor).item()
            val_acc = acc_metric(val_preds_tensor, val_labels_tensor).item()
            pbar.update(1);pbar.refresh()

            del train_preds_tensor, train_labels_tensor #, val_preds_tensor, val_labels_tensor

            # Logs
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            train_f1_history.append(train_f1)
            val_f1_history.append(val_f1)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            iou_history.append(iou)

            tqdm.write(
                f"{color('Epoch', 'bold', 'magenta')} {epoch + 1}/{hyperparams['epochs']} | "
                f"{color('Train', 'bold', 'cyan')}: "
                f"{color('loss')} {train_loss:.4f} {color('f1')} {train_f1:.4f} {color('acc')} {train_acc:.2%} | "
                f"{color('Val', 'bold', 'blue')}: "
                f"{color('loss', color='blue')} {val_loss:.4f} {color('f1', color='blue')} {val_f1:.4f} "
                f"{color('acc', color='blue')} {val_acc:.2%} {color('IoU', color='blue')} {iou:.4f}"
            )

            w = adaptive_loss.normalized_weights
            tqdm.write(f"Loss Weights → CE: {w[0]:.2f}, Dice: {w[1]:.2f}, Focal: {w[2]:.2f}")

            # Per-class F1 (same as before, just skip if not needed every epoch)
            per_class_f1 = f1_score(
                val_labels_tensor.cpu().numpy(),
                val_preds_tensor.cpu().numpy(),
                average=None,
                labels=list(range(1, NUM_CLASSES)),
                zero_division=0
            )

            del val_preds_tensor, val_labels_tensor

            cls_to_label = {v: k for k, v in label_to_cls.items()}
            per_class_names = [
                label_to_text.get(cls_to_label.get(i, -1), f"Class {i}")
                for i in range(1, NUM_CLASSES)
            ]

            tqdm.write(color("Classes with lowest val F1:", 'bold', 'yellow'))
            for i, (cls_name, f1) in enumerate(zip(per_class_names, per_class_f1), start=1):
                if f1 < 0.1:
                    tqdm.write(f"Cl.{i})  {cls_name}: {f1:.4f}")
            tqdm.write("")
            pbar.update(1); pbar.refresh()

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)

            with open(meta_file, "w") as f:
                f.write(f"Saved at Epoch: {epoch + 1}\n")
                f.write(f"Random Seed: {RANDOM_SEED}\n")
                f.write(f"Val F1: {val_f1:.4f}\n")
                f.write(f"Val Acc: {val_acc:.4f}\n")
                f.write(f"IoU: {iou:.4f}\n")
                f.write(f"LR: {hyperparams['lr']}, Batch Size: {hyperparams['batch_size']}\n")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch + 1}\n")
            f.write(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n")
            f.write(f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}\n")
            f.write(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}\n")
            f.write(f"IoU: {iou:.4f}\n")
            f.write(f"Best F1 So Far: {best_f1:.4f}\n")
            f.write("Per-Class F1:\n")
            for cls_name, f1 in zip(per_class_names, per_class_f1):
                f.write(f"{cls_name}: {f1:.4f}\n")
            f.write("-" * 40 + "\n")

        scheduler.step(val_f1)
        torch.cuda.empty_cache()

    # Plot training metrics
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_f1_history, label='Train F1')
    plt.plot(val_f1_history, label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(train_acc_history, label='Train Acc')
    plt.plot(val_acc_history, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'train_f1': train_f1_history,
        'val_f1': val_f1_history,
        'train_acc': train_acc_history,
        'val_acc': val_acc_history,
        'iou': iou_history
    }

def run_hyperparameter_experiment(train_dataset, val_dataset, device):
    # Experiment with different learning rates
    learning_rates = [1e-1, 5e-4, 1e-6]
    results = {}

    for lr in learning_rates:
        print(f"\nRunning experiment with LR={lr}")

        model = UNetResNet(encoder_depth=101, num_classes=NUM_CLASSES).to(device)
        hyperparams = {
            'batch_size': 12,
            'lr': lr,
            'epochs': 30
        }
        metrics = train_model(model, train_dataset, val_dataset, device,
                              hyperparams, output_dir=run_dir, early_patience=3)
        results[lr] = metrics

    print("Finished hyperparameter experiment.")
    import json

    # Save F1 and Accuracy histories to JSON (train + val)
    summary = {}
    for lr, metrics in results.items():
        summary[str(lr)] = {
            "train_f1": metrics["train_f1"],
            "val_f1": metrics["val_f1"],
            "train_acc": metrics["train_acc"],
            "val_acc": metrics["val_acc"]
        }

    with open("lr_experiment_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    from itertools import cycle

    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    plt.figure(figsize=(10, 6))
    for lr, metrics in results.items():
        c = next(colors)
        plt.plot(metrics['train_f1'], label=f'Train F1 (LR={lr})', color=c, alpha=0.7)
        plt.plot(metrics['val_f1'], label=f'Val F1 (LR={lr})', color=c)

    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Train/Val F1 Score for Different Learning Rates')
    plt.legend()
    plt.grid(True)
    plt.savefig('lr_comparison.png')
    plt.close()

    return results




def visualize_prediction(model, dataset, device, idx=0):
    model.eval()
    image, label, mask = dataset[idx]

    # Convert to tensor if needed
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    image_tensor = image.unsqueeze(0).to(device)  # (1, C, H, W)

    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    mismatches = (pred != label) & (mask != 0)
    num_mismatched = mismatches.sum()
    total_valid = (mask != 0).sum()
    accuracy = 100 * (1 - num_mismatched / total_valid)

    print(f"Mismatched {num_mismatched} pixels out of {total_valid} total pixels.")
    print(f"Patch Accuracy: {accuracy:.2f}%")

    image_np = image[:3].permute(1, 2, 0).numpy()  # Show first 3 channels

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image_np / image_np.max())
    axs[0].set_title("Input (first 3 bands)")
    axs[1].imshow(label, cmap='tab20')
    axs[1].set_title("Ground Truth")
    axs[2].imshow(pred, cmap='tab20')
    axs[2].set_title(f"Prediction\nAcc: {accuracy:.2f}% ({total_valid - num_mismatched}/{total_valid})")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("sample_prediction.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Load UNet Model for Sentinel2 Segmentation")
    parser.add_argument("--load", action="store_true", default=False, help="Load a model checkpoint instead of training")
    parser.add_argument("--experiment", action="store_true", default=False, help="Run experiment for hyperparameter tuning")
    parser.add_argument("--n_patches", type=int, default=5, help="Number of random patches to visualize")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"DEVICE: {device}")

    # Load datasets
    train_dataset = Sentinel2Dataset(
        split_txt=SPLIT_DIR / "train.txt",
        patch_dir=PATCH_DIR,
        transform=get_training_augmentations(seed=RANDOM_SEED),
        downsample_size=256
    )
    val_dataset = Sentinel2Dataset(
        split_txt=SPLIT_DIR / "val.txt",
        patch_dir=PATCH_DIR,
        transform=None,
        downsample_size=256
    )

    model = UNetResNet(encoder_depth=101, num_classes=NUM_CLASSES).to(device)

    if args.load:
        root = tk.Tk()
        root.withdraw()
        model_path = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select model checkpoint",
            filetypes=(("PyTorch checkpoint", "*.pth"), ("All files", "*.*"))
        )
        root.destroy()
        if model_path and os.path.exists(model_path):
            print("Loading model from checkpoint...")
            model.load_state_dict(torch.load(model_path))
        else:
            exit(1)
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("checkpoints", run_id)
        os.makedirs(run_dir, exist_ok=True)

        if not args.experiment:
            default_hyperparams = {
                'batch_size': 6,
                'lr': 1e-2,
                'epochs': 30
            }
            print(f"Training model from scratch with LR: {default_hyperparams['lr']} & batch size: {default_hyperparams['batch_size']} . . .")
            metrics = train_model(model, train_dataset, val_dataset, device,
                                  default_hyperparams, output_dir=run_dir, early_patience=3)
        else:
            run_hyperparameter_experiment(train_dataset, val_dataset, device)

    indices = random.sample(range(len(val_dataset)), 5)
    for idx in indices:
        print(f"\n=== Patch {idx} ===")
        visualize_prediction(model, val_dataset, device, idx=idx)