#!/usr/bin/env python3
"""
LocalML finetune CLI
Command-line interface for model management and training
"""

import click
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sentinel.cli.commands import model_group, dataset_group, train_group


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    LocalML finetune CLI

    Fast local fine-tuning with built-in models (FasterRCNN, CNN, CLIP)

    Examples:
        sentinel model import --path ./my_model.pth --name custom_resnet
        sentinel dataset prepare --path ./images --split 0.8 0.1 0.1
        sentinel train --model custom_resnet --dataset my_dataset --epochs 10
    """
    pass


# Add command groups
cli.add_command(model_group)
cli.add_command(dataset_group)
cli.add_command(train_group)


if __name__ == "__main__":
    cli()
