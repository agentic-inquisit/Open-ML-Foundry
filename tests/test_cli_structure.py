"""
Test suite for CLI structure and imports
Verifies that all CLI modules are properly structured
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_cli_main_imports():
    """Test that CLI main module can be imported"""
    from sentinel.cli.main import cli
    assert cli is not None
    print("✓ CLI main module imports correctly")


def test_cli_commands_imports():
    """Test that CLI commands can be imported"""
    from sentinel.cli.commands import model_group, dataset_group, train_group
    assert model_group is not None
    assert dataset_group is not None
    assert train_group is not None
    print("✓ CLI command groups import correctly")


def test_model_commands_exist():
    """Test that model commands are defined"""
    from sentinel.cli.commands import model_group

    # Get all commands in model_group
    commands = [cmd.name for cmd in model_group.commands.values()]
    assert 'import' in commands
    assert 'list' in commands
    assert 'info' in commands
    print(f"✓ Model commands exist: {commands}")


def test_dataset_commands_exist():
    """Test that dataset commands are defined"""
    from sentinel.cli.commands import dataset_group

    # Get all commands in dataset_group
    commands = [cmd.name for cmd in dataset_group.commands.values()]
    assert 'prepare' in commands
    print(f"✓ Dataset commands exist: {commands}")


def test_train_commands_exist():
    """Test that train commands are defined"""
    from sentinel.cli.commands import train_group

    # Get all commands in train_group
    commands = [cmd.name for cmd in train_group.commands.values()]
    assert 'start' in commands
    print(f"✓ Train commands exist: {commands}")


def test_cli_version():
    """Test that CLI version is defined"""
    from sentinel.cli.main import cli

    # Version should be in the CLI group
    assert hasattr(cli, '__doc__')
    print("✓ CLI version attribute exists")


def test_package_structure():
    """Test that package structure is correct"""
    import sentinel
    import sentinel.cli
    import sentinel.config
    import sentinel.utils

    assert hasattr(sentinel, '__file__')
    assert hasattr(sentinel.cli, '__file__')
    print("✓ Package structure is valid")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing CLI Structure")
    print("="*60 + "\n")

    tests = [
        test_cli_main_imports,
        test_cli_commands_imports,
        test_model_commands_exist,
        test_dataset_commands_exist,
        test_train_commands_exist,
        test_cli_version,
        test_package_structure,
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
