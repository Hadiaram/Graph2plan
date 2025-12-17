"""
Create train.txt and test.txt files for your custom dataset

This script reads your data.mat file and creates train/test splits
based on floor plan names.

Usage:
    python create_train_test_split.py --train-ratio 0.85
"""

import argparse
import numpy as np
import scipy.io as sio
from pathlib import Path
from config import data_path


def create_train_test_split(train_ratio=0.85, seed=42):
    """
    Create train.txt and test.txt files

    Args:
        train_ratio: Percentage of data to use for training (default: 0.85)
        seed: Random seed for reproducibility
    """
    print(f"Loading data from: {data_path}")

    # Load the mat file
    data = sio.loadmat(data_path, squeeze_me=True, struct_as_record=False)['data']

    # Get all floor plan names
    if hasattr(data, '__len__'):
        # Array of items
        names = [item.name if hasattr(item, 'name') else str(i)
                 for i, item in enumerate(data)]
    else:
        # Single item
        names = [data.name if hasattr(data, 'name') else '0']

    n_total = len(names)
    print(f"Total floor plans: {n_total}")

    # Shuffle names
    np.random.seed(seed)
    indices = np.random.permutation(n_total)

    # Split into train and test
    n_train = int(n_total * train_ratio)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_names = [names[i] for i in train_indices]
    test_names = [names[i] for i in test_indices]

    print(f"Training samples: {len(train_names)}")
    print(f"Test samples: {len(test_names)}")

    # Create data directory if it doesn't exist
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)

    # Write train.txt
    train_file = data_dir / 'train.txt'
    with open(train_file, 'w') as f:
        f.write('\n'.join(train_names))
    print(f"\nCreated: {train_file}")

    # Write test.txt
    test_file = data_dir / 'test.txt'
    with open(test_file, 'w') as f:
        f.write('\n'.join(test_names))
    print(f"Created: {test_file}")

    # Print first few names as sample
    print(f"\nFirst 5 training names:")
    for name in train_names[:5]:
        print(f"  {name}")

    print(f"\nFirst 5 test names:")
    for name in test_names[:5]:
        print(f"  {name}")

    print("\n✓ Train/test split complete!")
    print(f"  Train: {len(train_names)} ({train_ratio*100:.1f}%)")
    print(f"  Test: {len(test_names)} ({(1-train_ratio)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Create train.txt and test.txt files for custom dataset'
    )
    parser.add_argument('--train-ratio', type=float, default=0.85,
                       help='Ratio of data to use for training (default: 0.85)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    if args.train_ratio <= 0 or args.train_ratio >= 1:
        print("Error: train-ratio must be between 0 and 1")
        return

    create_train_test_split(args.train_ratio, args.seed)


if __name__ == '__main__':
    main()
