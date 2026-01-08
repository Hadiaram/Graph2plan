"""
Simple Dataset Reduction - Reduce train.txt and test.txt to 1000 total samples
Target: 800 train + 200 test = 1,000 total
"""

import random

# Configuration
TARGET_TRAIN_SIZE = 800
TARGET_TEST_SIZE = 200
RANDOM_SEED = 42

# Read current files
with open('DataPreparation/data/train.txt', 'r') as f:
    train_ids = [line.strip() for line in f if line.strip()]

with open('DataPreparation/data/test.txt', 'r') as f:
    test_ids = [line.strip() for line in f if line.strip()]

print(f"Original: {len(train_ids)} train, {len(test_ids)} test = {len(train_ids) + len(test_ids)} total")

# Sample
random.seed(RANDOM_SEED)
sampled_train = random.sample(train_ids, TARGET_TRAIN_SIZE)
sampled_test = random.sample(test_ids, TARGET_TEST_SIZE)

# Write back
with open('DataPreparation/data/train.txt', 'w') as f:
    for id in sampled_train:
        f.write(id + '\n')

with open('DataPreparation/data/test.txt', 'w') as f:
    for id in sampled_test:
        f.write(id + '\n')

print(f"New:      {len(sampled_train)} train, {len(sampled_test)} test = {len(sampled_train) + len(sampled_test)} total")
print("✅ Done! train.txt and test.txt updated.")
