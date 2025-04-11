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

from v_prepare_training_data import Sentinel2Dataset, get_training_augmentations, PATCH_DIR, SPLIT_DIR
from vi2_sentinel2_unet import UNetResNet, DiceLoss
import os


def color(text, style='bold', color='cyan'):
    styles = {'bold': '1', 'dim': '2', 'normal': '22'}
    colors = {'cyan': '36', 'magenta': '35', 'yellow': '33', 'blue': '34'}
    return f"\033[{styles[style]};{colors[color]}m{text}\033[0m"


def train_model(model, train_dataset, val_dataset, device, hyperparams, output_dir, patience=3):
    model_path = os.path.join(output_dir, "best_model.pth")
    log_file = os.path.join(output_dir, "training_log.txt")
    meta_file = os.path.join(output_dir, "best_model_meta.txt")
    plot_path = os.path.join(output_dir, "training_metrics.png")

    epochs_no_improve = 0

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparams['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparams['batch_size'],
        shuffle=False,
        num_workers=4
    )

    # Loss and optimizer
    # criterion = nn.CrossEntropyLoss(ignore_index=0)

    ce_criterion = nn.CrossEntropyLoss(ignore_index=0)
    ce_weight = 0.4
    dice_criterion = DiceLoss()
    dice_weight = 1-ce_weight

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
            loss = ce_weight * ce_criterion(outputs, labels) + dice_weight * dice_criterion(outputs, labels, masks)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            valid_mask = masks.bool()
            train_preds.extend(preds[valid_mask].cpu().numpy())
            train_labels.extend(labels[valid_mask].cpu().numpy())

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
                loss = ce_weight * ce_criterion(outputs, labels) + dice_weight * dice_criterion(outputs, labels, masks)
                epoch_val_loss += loss.item() * images.size(0)

                preds = torch.argmax(outputs, dim=1)
                valid_mask = masks.bool()
                val_preds.extend(preds[valid_mask].cpu().numpy())
                val_labels.extend(labels[valid_mask].cpu().numpy())

        # Metrics
        train_loss = epoch_train_loss / len(train_loader.dataset)
        val_loss = epoch_val_loss / len(val_loader.dataset)

        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
        train_acc = np.mean(np.array(train_labels) == np.array(train_preds))
        val_acc = np.mean(np.array(val_labels) == np.array(val_preds))
        iou = jaccard_score(val_labels, val_preds, average='weighted', zero_division=0)

        # Logs
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_f1_history.append(train_f1)
        val_f1_history.append(val_f1)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        iou_history.append(iou)

        print(
            f"\n{color('Epoch', 'bold', 'magenta')} {epoch + 1}/{hyperparams['epochs']} | "
            f"{color('Train', 'bold', 'cyan')}: "
            f"{color('loss')} {train_loss:.4f} {color('f1')} {train_f1:.4f} {color('acc')} {train_acc:.2%} | "
            f"{color('Val', 'bold', 'blue')}: "
            f"{color('loss', color='blue')} {val_loss:.4f} {color('f1', color='blue')} {val_f1:.4f} "
            f"{color('acc', color='blue')} {val_acc:.2%} {color('IoU', color='blue')} {iou:.4f}"
        )
        num_classes = model.final_conv[-1].out_channels
        per_class_f1 = f1_score(val_labels, val_preds, average=None, labels=list(range(1, num_classes)), zero_division=0)
        per_class_names = [f"Class {i}" for i in range(1, num_classes)]
        print(color("Per-Class F1:", 'bold', 'yellow'))
        for cls_name, f1 in zip(per_class_names, per_class_f1):
            print(f"  {cls_name}: {f1:.4f}")
        print("\n")


        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)

            with open(meta_file, "w") as f:
                f.write(f"Saved at Epoch: {epoch + 1}\n")
                f.write(f"Val F1: {val_f1:.4f}\n")
                f.write(f"Val Acc: {val_acc:.4f}\n")
                f.write(f"IoU: {iou:.4f}\n")
                f.write(f"LR: {hyperparams['lr']}, Batch Size: {hyperparams['batch_size']}\n")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
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
    learning_rates = [1e-3, 5e-4, 1e-4]
    results = {}

    for lr in learning_rates:
        print(f"\nRunning experiment with LR={lr}")
        model = UNetResNet(encoder_depth=50, num_classes=10).to(device)
        hyperparams = {
            'batch_size': 8,
            'lr': lr,
            'epochs': 20
        }

        metrics = train_model(model, train_dataset, val_dataset, device, hyperparams)
        results[lr] = metrics

    # Plot comparison
    plt.figure(figsize=(10, 6))
    for lr, metrics in results.items():
        plt.plot(metrics['f1'], label=f'LR={lr}')

    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score for Different Learning Rates')
    plt.legend()
    plt.grid(True)
    plt.savefig('lr_comparison.png')
    plt.close()

    return results


import matplotlib.pyplot as plt


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
    parser.add_argument("--n_patches", type=int, default=5, help="Number of random patches to visualize")
    args = parser.parse_args()

    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"DEVICE: {device}")

    # Load datasets
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

    model = UNetResNet(encoder_depth=101, num_classes=10).to(device)

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
        print("Training model from scratch...")

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("checkpoints", run_id)
        os.makedirs(run_dir, exist_ok=True)

        default_hyperparams = {
            'batch_size': 8,
            'lr': 1e-4,
            'epochs': 30
        }
        metrics = train_model(model, train_dataset, val_dataset, device, default_hyperparams, output_dir=run_dir)

    indices = random.sample(range(len(val_dataset)), 5)
    for idx in indices:
        print(f"\n=== Patch {idx} ===")
        visualize_prediction(model, val_dataset, device, idx=idx)