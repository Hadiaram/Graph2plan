"""
Quick check: Are there any room type indices in the data that aren't in our vocabulary?
"""
import scipy.io as sio
import numpy as np
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from model.utils import get_vocab

print("=" * 70)
print("CHECKING FOR UNMAPPED ROOM TYPE INDICES")
print("=" * 70)

# Get vocabulary
vocab = get_vocab()
valid_data_indices = set(vocab['data_idx_to_model_idx'].keys())
num_model_classes = len(vocab['object_idx_to_name'])

print(f"\n1. Vocabulary Info:")
print(f"   Model has {num_model_classes} classes (indices 0-{num_model_classes-1})")
print(f"   Valid data indices: {sorted(valid_data_indices)}")
print(f"   Mapping: {vocab['data_idx_to_model_idx']}")

# Check training data
print(f"\n2. Checking data/data_train.mat...")
data = sio.loadmat('./data/data_train.mat', squeeze_me=True, struct_as_record=False)

all_indices = set()
problem_indices = set()

for sample in data['data']:
    # Extract room types - handle both old and new format
    if hasattr(sample, 'rType'):
        room_types = sample.rType if isinstance(sample.rType, np.ndarray) else [sample.rType]
    elif hasattr(sample, 'box'):
        room_types = sample.box[:, 4].astype(int)
    else:
        continue
    
    for idx in room_types:
        all_indices.add(int(idx))
        if int(idx) not in valid_data_indices:
            problem_indices.add(int(idx))

print(f"\n3. Results:")
print(f"   All unique indices in data: {sorted(all_indices)}")

if problem_indices:
    print(f"\n   ❌ PROBLEM FOUND!")
    print(f"   These indices are in data but NOT in vocabulary mapping:")
    print(f"   {sorted(problem_indices)}")
    print(f"\n   These indices will cause 'srcIndex < srcSelectDimSize' errors!")
    print(f"\n   Solution: Add these to vocabulary mapping in model/utils.py")
else:
    print(f"\n   ✅ All data indices are in vocabulary mapping")

# Also check if any mapped indices would be out of bounds
print(f"\n4. Checking mapped indices...")
for data_idx in all_indices:
    if data_idx in vocab['data_idx_to_model_idx']:
        model_idx = vocab['data_idx_to_model_idx'][data_idx]
        if model_idx >= num_model_classes:
            print(f"   ❌ ERROR: Data index {data_idx} maps to model index {model_idx}")
            print(f"      But model only has {num_model_classes} classes!")
        else:
            print(f"   ✅ Data {data_idx} → Model {model_idx} (valid)")
    else:
        print(f"   ❌ Data index {data_idx} has NO mapping!")

print("\n" + "=" * 70)
