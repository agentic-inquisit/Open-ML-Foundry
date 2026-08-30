#!/usr/bin/env python3
"""
Image Classification Fine-Tuning Example

Fine-tune a ResNet50 model on a custom image dataset.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ImageFolder
from tqdm import tqdm


class ImageClassifier:
    """Fine-tune image classification model."""

    def __init__(self, model_name: str = "resnet50", num_classes: int = 10):
        """Initialize classifier.

        Args:
            model_name: Model architecture (resnet50, resnet18, etc)
            num_classes: Number of output classes
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained model
        if model_name == "resnet50":
            import torchvision.models as models

            self.model = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Replace classification head
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.CrossEntropyLoss()
        self.history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average loss for epoch
        """
        self.model.train()
        total_loss = 0.0

        for images, labels in tqdm(train_loader, desc="Training"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        self.history["train_loss"].append(avg_loss)
        return avg_loss

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        self.history["val_loss"].append(avg_loss)
        self.history["val_accuracy"].append(accuracy)

        return avg_loss, accuracy

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
    ) -> Dict:
        """Train model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs

        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print("Image Classification Fine-Tuning")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            train_loss = self.train_epoch(train_loader)
            val_loss, val_accuracy = self.validate(val_loader)

            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.2f}%")

        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}\n")

        return self.history

    def save(self, path: str) -> None:
        """Save model.

        Args:
            path: Path to save model
        """
        torch.save(self.model.state_dict(), path)
        print(f"✓ Model saved to {path}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Image Classification Fine-Tuning")
    parser.add_argument("--data", required=True, help="Path to dataset directory")
    parser.add_argument("--model", default="resnet50", help="Model name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--output", default="model.pth", help="Output model path")

    args = parser.parse_args()

    # Data transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load datasets
    train_dataset = ImageFolder(f"{args.data}/train", transform=transform)
    val_dataset = ImageFolder(f"{args.data}/val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Train model
    classifier = ImageClassifier(
        model_name=args.model,
        num_classes=len(train_dataset.classes),
    )

    history = classifier.fit(train_loader, val_loader, epochs=args.epochs)

    # Save model
    classifier.save(args.output)

    # Save history
    history_path = Path(args.output).stem + "_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved to {history_path}")


if __name__ == "__main__":
    main()
