"""
Reduce Dataset Size - Sample 1000 Floor Plans from Full Dataset
Original: 14,446 train + 2,550 test = 16,996 total
Target: 800 train + 200 test = 1,000 total
"""

import random
import shutil
import os
from datetime import datetime

# Configuration
TRAIN_FILE = 'DataPreparation/data/train.txt'
TEST_FILE = 'DataPreparation/data/test.txt'
BACKUP_DIR = 'DataPreparation/data/backup'
TARGET_TRAIN_SIZE = 800
TARGET_TEST_SIZE = 200
RANDOM_SEED = 42  # For reproducibility

def backup_files():
    """Backup original files with timestamp"""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    train_backup = f'{BACKUP_DIR}/train_full_{timestamp}.txt'
    test_backup = f'{BACKUP_DIR}/test_full_{timestamp}.txt'
    
    shutil.copy(TRAIN_FILE, train_backup)
    shutil.copy(TEST_FILE, test_backup)
    
    print(f"✅ Backed up original files:")
    print(f"   {train_backup}")
    print(f"   {test_backup}")
    print()

def sample_and_save(input_file, output_file, target_size, seed):
    """Randomly sample target_size lines from input_file"""
    # Read all lines
    with open(input_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    original_size = len(lines)
    
    # Random sample
    random.seed(seed)
    if target_size >= original_size:
        sampled_lines = lines  # Keep all if target is larger
    else:
        sampled_lines = random.sample(lines, target_size)
    
    # Write sampled lines
    with open(output_file, 'w') as f:
        for line in sampled_lines:
            f.write(line + '\n')
    
    return original_size, len(sampled_lines)

def main():
    print("=" * 60)
    print("Dataset Reduction: 16,996 → 1,000 Floor Plans")
    print("=" * 60)
    print()
    
    # Step 1: Backup
    print("Step 1: Backing up original files...")
    backup_files()
    
    # Step 2: Sample training set
    print("Step 2: Sampling training set...")
    train_orig, train_new = sample_and_save(
        TRAIN_FILE, TRAIN_FILE, TARGET_TRAIN_SIZE, RANDOM_SEED
    )
    print(f"   Original: {train_orig:,} samples")
    print(f"   Sampled:  {train_new:,} samples")
    print(f"   Reduction: {train_orig - train_new:,} samples removed")
    print()
    
    # Step 3: Sample test set
    print("Step 3: Sampling test set...")
    test_orig, test_new = sample_and_save(
        TEST_FILE, TEST_FILE, TARGET_TEST_SIZE, RANDOM_SEED
    )
    print(f"   Original: {test_orig:,} samples")
    print(f"   Sampled:  {test_new:,} samples")
    print(f"   Reduction: {test_orig - test_new:,} samples removed")
    print()
    
    # Summary
    print("=" * 60)
    print("✅ Dataset Reduction Complete!")
    print("=" * 60)
    print(f"Total Original: {train_orig + test_orig:,} floor plans")
    print(f"Total New:      {train_new + test_new:,} floor plans")
    print(f"Total Removed:  {(train_orig + test_orig) - (train_new + test_new):,} floor plans")
    print()
    print(f"New Split: {train_new} train / {test_new} test")
    print(f"Ratio: {train_new/(train_new+test_new)*100:.1f}% train / {test_new/(train_new+test_new)*100:.1f}% test")
    print()
    print("Original files backed up in: DataPreparation/data/backup/")
    print()
    print("⚠️ IMPORTANT: You also need to regenerate the .mat files:")
    print("   1. Run: python DataPreparation/2.data_train_converted.py")
    print("   2. Run: python DataPreparation/5.data_test_converted.py")
    print("   3. Update Network/data/*.mat files")

if __name__ == '__main__':
    main()
