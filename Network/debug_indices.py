"""
Debug script to find which operation is causing the index error
"""
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import scipy.io as sio
import numpy as np
from model.utils import get_vocab
from model.floorplan import FloorPlanDataset

print("=" * 70)
print("DEBUGGING INDEX OUT OF BOUNDS ERROR")
print("=" * 70)

# Load data
print("\n1. Loading data...")
data = sio.loadmat('./data/data_train.mat', squeeze_me=True, struct_as_record=False)

# Get vocabulary
vocab = get_vocab()
print(f"\n2. Vocabulary Info:")
print(f"   Num classes: {len(vocab['object_idx_to_name'])}")
print(f"   Index mapping: {vocab['data_idx_to_model_idx']}")

# Create dataset
print(f"\n3. Creating dataset...")
dataset = FloorPlanDataset(data, vocab, 'train')

print(f"\n4. Checking first few samples...")
for i in range(min(5, len(dataset))):
    try:
        sample = dataset[i]
        objs = sample['objs']
        
        print(f"\n   Sample {i}:")
        print(f"     Raw objs shape: {objs.shape}")
        print(f"     Objs values: {objs}")
        print(f"     Min: {objs.min()}, Max: {objs.max()}")
        
        # Check if any index is >= num_classes
        num_classes = len(vocab['object_idx_to_name'])
        if objs.max() >= num_classes:
            print(f"     ❌ ERROR: Found index {objs.max()} >= num_classes ({num_classes})")
            print(f"     This will cause embedding lookup failure!")
            
            # Find which original indices weren't remapped
            print(f"\n     Investigating unmapped indices...")
            for obj_idx in objs:
                if obj_idx >= num_classes:
                    print(f"       Index {obj_idx} is out of bounds!")
        else:
            print(f"     ✅ All indices valid (0-{num_classes-1})")
            
    except Exception as e:
        print(f"   ❌ Error loading sample {i}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
