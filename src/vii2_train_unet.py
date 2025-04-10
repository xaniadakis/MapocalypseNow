import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, jaccard_score
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random

from v_prepare_training_data import Sentinel2Dataset, get_training_augmentations, PATCH_DIR, SPLIT_DIR
from vi2_sentinel2_unet import UNetResNet
import os


def train_model(model, train_dataset, val_dataset, device, hyperparams, patience=3):
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
    criterion = nn.CrossEntropyLoss(ignore_index=0)
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

        for images, labels, masks in train_loader:
            images = images.to(device)
            labels = labels.to(device).long()
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
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
            for images, labels, masks in val_loader:
                images = images.to(device)
                labels = labels.to(device).long()
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item() * images.size(0)

                preds = torch.argmax(outputs, dim=1)
                valid_mask = masks.bool()
                val_preds.extend(preds[valid_mask].cpu().numpy())
                val_labels.extend(labels[valid_mask].cpu().numpy())

        # Metrics
        train_loss = epoch_train_loss / len(train_loader.dataset)
        val_loss = epoch_val_loss / len(val_loader.dataset)

        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        train_acc = np.mean(np.array(train_labels) == np.array(train_preds))
        val_acc = np.mean(np.array(val_labels) == np.array(val_preds))
        iou = jaccard_score(val_labels, val_preds, average='weighted')

        # Logs
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_f1_history.append(train_f1)
        val_f1_history.append(val_f1)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        iou_history.append(iou)

        print(f"Epoch {epoch + 1}/{hyperparams['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print(f"IoU: {iou:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            model_path = os.path.join("checkpoints", "best_model.pth")
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        scheduler.step(val_f1)

    # Plot
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
    plt.savefig('training_metrics.png')
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

    os.makedirs("checkpoints", exist_ok=True)
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"DEVICE: {device}")

    # Load datasets (using your previous implementation)
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

    model = UNetResNet(encoder_depth=50, num_classes=10).to(device)

    load_model = True  # set to False to disable loading
    model_path = os.path.join("checkpoints", "best_model.pth")

    if load_model and os.path.exists(model_path):
        print("Loading model from checkpoint...")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Training model from scratch...")

        # Train with default hyperparameters
        default_hyperparams = {
            'batch_size': 8,
            'lr': 1e-4,
            'epochs': 30
        }
        metrics = train_model(model, train_dataset, val_dataset, device, default_hyperparams)


    indices = random.sample(range(len(val_dataset)), 5)
    for idx in indices:
        print(f"\n=== Patch {idx} ===")
        visualize_prediction(model, val_dataset, device, idx=idx)

        # Run hyperparameter experiment
        # lr_results = run_hyperparameter_experiment(train_dataset, val_dataset, device)