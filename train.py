"""
Train the SimpleCNN on the Fake Currency Detection dataset.

Reads images directly from data/Real and data/Fake (with denomination
subfolders) and performs an 80/20 train/test split internally - no manual
file moving required.

Usage:
    python train.py
    python train.py --epochs 25 --batch_size 32 --lr 0.001

Saves the best model (based on test accuracy) to:
    saved_models/best_model.pth
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn_model import SimpleCNN
from utils.dataset import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train Fake Currency Detection CNN")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory (contains Real/ and Fake/)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=128, help="Image size (square)")
    parser.add_argument("--test_split", type=float, default=0.2, help="Fraction of data held out for testing")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Where to save model checkpoints")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Data (80/20 split happens automatically inside get_dataloaders) ----
    train_loader, test_loader, class_names = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, test_split=args.test_split
    )
    print(f"Class-to-index mapping: {dict((c, i) for i, c in enumerate(class_names))}")

    # ---- Model / Loss / Optimizer ----
    model = SimpleCNN(num_classes=len(class_names), img_size=args.img_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_test_acc = 0.0

    print("\nStarting training...\n")
    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step(test_acc)

        elapsed = time.time() - start
        print(f"Epoch [{epoch}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} | "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "img_size": args.img_size,
                "test_acc": test_acc,
            }
            save_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(checkpoint, save_path)
            print(f"  -> New best model saved (test_acc={test_acc:.4f}) at {save_path}")

    print(f"\nBest test accuracy achieved: {best_test_acc:.4f}")
    print(f"Model saved at: {os.path.join(args.save_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()
