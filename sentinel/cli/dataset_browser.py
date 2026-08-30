"""
Dataset Browser Module
Provides advanced dataset inspection, validation, and visualization
"""

from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime


class DatasetBrowser:
    """Browse, validate, and analyze datasets"""

    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}

    def __init__(self, path):
        """Initialize browser with dataset path"""
        self.path = Path(path)
        self.images = []
        self.structure = None
        self.classes = defaultdict(list)
        self.metadata = {
            "path": str(self.path),
            "scanned_at": datetime.now().isoformat(),
            "structure": None,
            "total_images": 0,
            "classes": {},
            "warnings": []
        }

    def scan(self):
        """Scan dataset and detect structure"""
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.path}")

        # Find all images
        self.images = [
            f for f in self.path.rglob('*')
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_FORMATS
        ]

        if not self.images:
            raise ValueError(f"No images found in {self.path}")

        # Detect structure
        self._detect_structure()

        # Analyze classes
        self._analyze_classes()

        # Validate dataset
        self._validate()

        return self

    def _detect_structure(self):
        """Detect dataset structure (flat vs class-based)"""
        # Check if all images are in root
        root_images = [
            f for f in self.path.iterdir()
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_FORMATS
        ]

        if len(root_images) == len(self.images):
            self.structure = "flat"
        else:
            # Check if subdirectories are classes
            subdirs = [d for d in self.path.iterdir() if d.is_dir()]
            if subdirs:
                self.structure = "class_folders"
            else:
                self.structure = "nested"

    def _analyze_classes(self):
        """Analyze class distribution"""
        if self.structure == "class_folders":
            for img in self.images:
                # Get class from parent folder
                class_name = img.parent.name
                if class_name != self.path.name:  # Skip root folder
                    self.classes[class_name].append(img)

        elif self.structure == "flat":
            # Single class for flat structure
            self.classes["all_images"] = self.images

        else:
            # Nested - try to find class in path
            for img in self.images:
                # Use first subdirectory as class
                rel_path = img.relative_to(self.path)
                if len(rel_path.parts) > 1:
                    class_name = rel_path.parts[0]
                else:
                    class_name = "uncategorized"
                self.classes[class_name].append(img)

    def _validate(self):
        """Validate dataset and collect warnings"""
        warnings = []

        # Check for class imbalance
        if self.structure in ["class_folders", "nested"] and len(self.classes) > 1:
            counts = {cls: len(imgs) for cls, imgs in self.classes.items()}
            max_count = max(counts.values())
            min_count = min(counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

            if imbalance_ratio > 5:
                warnings.append(
                    f"⚠️ Class imbalance detected (ratio {imbalance_ratio:.1f}x). "
                    f"Consider balancing: {min_count} to {max_count} images per class"
                )

        # Check for insufficient data
        if len(self.images) < 10:
            warnings.append(
                f"⚠️ Only {len(self.images)} images. Recommend 50+ for training"
            )

        # Check for classes with very few images
        for cls, imgs in self.classes.items():
            if len(imgs) < 3:
                warnings.append(
                    f"⚠️ Class '{cls}' has only {len(imgs)} images. Recommend 10+ per class"
                )

        # Check for duplicate filenames
        filenames = [img.name for img in self.images]
        if len(filenames) != len(set(filenames)):
            warnings.append(
                "⚠️ Duplicate filenames found across different folders"
            )

        self.metadata["warnings"] = warnings

    def get_summary(self):
        """Get dataset summary"""
        return {
            "total_images": len(self.images),
            "structure": self.structure,
            "num_classes": len(self.classes),
            "classes": {cls: len(imgs) for cls, imgs in self.classes.items()},
            "warnings": self.metadata["warnings"]
        }

    def get_class_distribution(self):
        """Get class distribution as percentages"""
        total = len(self.images)
        if total == 0:
            return {}

        return {
            cls: {
                "count": len(imgs),
                "percentage": (len(imgs) / total) * 100
            }
            for cls, imgs in sorted(
                self.classes.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )
        }

    def get_split(self, train_ratio=0.8, val_ratio=0.1):
        """Get train/val/test split"""
        test_ratio = 1.0 - train_ratio - val_ratio

        splits = {
            "train": {"images": [], "count": 0, "percentage": train_ratio * 100},
            "val": {"images": [], "count": 0, "percentage": val_ratio * 100},
            "test": {"images": [], "count": 0, "percentage": test_ratio * 100}
        }

        total = len(self.images)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)

        for i, img in enumerate(self.images):
            if i < train_count:
                splits["train"]["images"].append(str(img))
                splits["train"]["count"] += 1
            elif i < train_count + val_count:
                splits["val"]["images"].append(str(img))
                splits["val"]["count"] += 1
            else:
                splits["test"]["images"].append(str(img))
                splits["test"]["count"] += 1

        return splits

    def get_sample_images(self, n=5):
        """Get first N sample images"""
        return [str(img) for img in self.images[:n]]

    def export_metadata(self, output_file=None):
        """Export dataset metadata to JSON"""
        metadata = {
            "path": str(self.path),
            "scanned_at": datetime.now().isoformat(),
            "structure": self.structure,
            "summary": self.get_summary(),
            "class_distribution": self.get_class_distribution(),
            "samples": self.get_sample_images(5)
        }

        if output_file:
            output_path = Path(output_file)
            output_path.write_text(json.dumps(metadata, indent=2))
            return str(output_path)

        return metadata

    def print_report(self):
        """Print a formatted report of the dataset"""
        summary = self.get_summary()
        distribution = self.get_class_distribution()

        print(f"\n{'='*60}")
        print(f"📊 Dataset Report: {self.path.name}")
        print(f"{'='*60}")

        print(f"\n📈 Summary:")
        print(f"   Total images: {summary['total_images']}")
        print(f"   Structure: {summary['structure'].replace('_', ' ').title()}")
        print(f"   Number of classes: {summary['num_classes']}")

        if summary['num_classes'] > 1:
            print(f"\n🏷️ Class Distribution:")
            for cls, stats in distribution.items():
                bar_length = int(stats['percentage'] / 5)
                bar = "█" * bar_length
                print(f"   {cls:20} {stats['count']:4d} images {stats['percentage']:5.1f}% {bar}")

        if summary['warnings']:
            print(f"\n⚠️ Warnings:")
            for warning in summary['warnings']:
                print(f"   {warning}")

        print(f"\n{'='*60}\n")

    def interactive_preview(self):
        """Interactive dataset preview (for future terminal UI)"""
        # Placeholder for Phase 3
        pass
