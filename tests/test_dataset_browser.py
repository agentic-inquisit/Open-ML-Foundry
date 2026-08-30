"""
Test suite for DatasetBrowser module
Verifies dataset scanning, validation, and analysis
"""

import sys
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentinel.cli.dataset_browser import DatasetBrowser


def create_test_dataset():
    """Create a temporary test dataset"""
    tmpdir = tempfile.mkdtemp(prefix="test_dataset_")
    tmppath = Path(tmpdir)

    # Create class folders with images
    classes = {
        "dogs": 5,
        "cats": 3,
        "birds": 2
    }

    for class_name, count in classes.items():
        class_dir = tmppath / class_name
        class_dir.mkdir()

        for i in range(count):
            # Create dummy image files
            img_file = class_dir / f"{class_name}_{i}.jpg"
            img_file.write_text(f"dummy image {i}")

    return tmppath


def test_dataset_browser_scan():
    """Test dataset browser can scan a dataset"""
    tmppath = create_test_dataset()
    try:
        browser = DatasetBrowser(str(tmppath))
        browser.scan()

        assert browser.images is not None
        assert len(browser.images) == 10
        print(f"✓ Dataset browser found {len(browser.images)} images")

    finally:
        shutil.rmtree(tmppath)


def test_structure_detection():
    """Test structure detection (class folders vs flat)"""
    tmppath = create_test_dataset()
    try:
        browser = DatasetBrowser(str(tmppath))
        browser.scan()

        assert browser.structure == "class_folders"
        print(f"✓ Structure detected as: {browser.structure}")

    finally:
        shutil.rmtree(tmppath)


def test_class_analysis():
    """Test class detection and counting"""
    tmppath = create_test_dataset()
    try:
        browser = DatasetBrowser(str(tmppath))
        browser.scan()

        assert len(browser.classes) == 3
        assert len(browser.classes["dogs"]) == 5
        assert len(browser.classes["cats"]) == 3
        assert len(browser.classes["birds"]) == 2
        print(f"✓ Classes detected: {list(browser.classes.keys())}")

    finally:
        shutil.rmtree(tmppath)


def test_summary_generation():
    """Test summary generation"""
    tmppath = create_test_dataset()
    try:
        browser = DatasetBrowser(str(tmppath))
        browser.scan()

        summary = browser.get_summary()

        assert summary["total_images"] == 10
        assert summary["structure"] == "class_folders"
        assert summary["num_classes"] == 3
        assert "dogs" in summary["classes"]
        print(f"✓ Summary generated: {summary['total_images']} images, {summary['num_classes']} classes")

    finally:
        shutil.rmtree(tmppath)


def test_class_distribution():
    """Test class distribution calculation"""
    tmppath = create_test_dataset()
    try:
        browser = DatasetBrowser(str(tmppath))
        browser.scan()

        distribution = browser.get_class_distribution()

        assert "dogs" in distribution
        assert distribution["dogs"]["count"] == 5
        assert distribution["dogs"]["percentage"] == 50.0
        print(f"✓ Distribution calculated: {distribution['dogs']['percentage']:.1f}% for dogs")

    finally:
        shutil.rmtree(tmppath)


def test_split_generation():
    """Test train/val/test split"""
    tmppath = create_test_dataset()
    try:
        browser = DatasetBrowser(str(tmppath))
        browser.scan()

        splits = browser.get_split(train_ratio=0.6, val_ratio=0.2)

        assert splits["train"]["count"] == 6
        assert splits["val"]["count"] == 2
        assert splits["test"]["count"] == 2
        print(f"✓ Splits generated: {splits['train']['count']} train, {splits['val']['count']} val, {splits['test']['count']} test")

    finally:
        shutil.rmtree(tmppath)


def test_validation():
    """Test dataset validation"""
    tmppath = create_test_dataset()
    try:
        browser = DatasetBrowser(str(tmppath))
        browser.scan()

        summary = browser.get_summary()

        # Should have warnings for imbalance (5 vs 2)
        assert len(summary["warnings"]) > 0
        print(f"✓ Validation warnings generated: {len(summary['warnings'])} warnings")

    finally:
        shutil.rmtree(tmppath)


def test_sample_images():
    """Test sample image retrieval"""
    tmppath = create_test_dataset()
    try:
        browser = DatasetBrowser(str(tmppath))
        browser.scan()

        samples = browser.get_sample_images(n=3)

        assert len(samples) == 3
        assert all(Path(s).exists() for s in samples)
        print(f"✓ Sample images retrieved: {len(samples)} samples")

    finally:
        shutil.rmtree(tmppath)


def test_flat_structure():
    """Test flat directory structure detection"""
    tmppath = Path(tempfile.mkdtemp(prefix="test_flat_"))
    try:
        # Create flat structure (all images in root)
        for i in range(5):
            img_file = tmppath / f"image_{i}.jpg"
            img_file.write_text(f"dummy image {i}")

        browser = DatasetBrowser(str(tmppath))
        browser.scan()

        assert browser.structure == "flat"
        assert len(browser.images) == 5
        print(f"✓ Flat structure detected with {len(browser.images)} images")

    finally:
        shutil.rmtree(tmppath)


def test_error_handling():
    """Test error handling for invalid paths"""
    try:
        browser = DatasetBrowser("/nonexistent/path")
        browser.scan()
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError:
        print("✓ FileNotFoundError raised for invalid path")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing DatasetBrowser Module")
    print("="*60 + "\n")

    tests = [
        test_dataset_browser_scan,
        test_structure_detection,
        test_class_analysis,
        test_summary_generation,
        test_class_distribution,
        test_split_generation,
        test_validation,
        test_sample_images,
        test_flat_structure,
        test_error_handling,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    sys.exit(0 if failed == 0 else 1)
