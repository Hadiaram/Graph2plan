"""
Verify that the updated vocabulary matches the data indices
"""
import sys
sys.path.insert(0, './model')
from utils import get_vocab
import scipy.io as sio
import numpy as np

print("=" * 80)
print("VOCABULARY VERIFICATION")
print("=" * 80)

# Get vocabulary
vocab = get_vocab()
print(f"\n1. Vocabulary Information:")
print(f"   - Number of object types: {len(vocab['object_idx_to_name'])}")
print(f"   - Object names: {vocab['object_idx_to_name']}")
print(f"   - Object name to index: {vocab['object_name_to_idx']}")

# Load data and check indices
print(f"\n2. Data Information:")
data = sio.loadmat('./data/data_train.mat', squeeze_me=True, struct_as_record=False)
all_objs = []
for sample in data['data'][0]:
    objs = sample[3][0]  # Index 3 is 'objs'
    all_objs.extend(objs)

unique_data_indices = np.unique(all_objs)
print(f"   - Unique indices in data: {unique_data_indices}")
print(f"   - Number of unique indices: {len(unique_data_indices)}")

# Check for mismatches
vocab_indices = set(vocab['object_name_to_idx'].values())
data_indices = set(unique_data_indices.tolist())

print(f"\n3. Compatibility Check:")
print(f"   - Vocabulary indices: {sorted(vocab_indices)}")
print(f"   - Data indices: {sorted(data_indices)}")

if data_indices.issubset(vocab_indices):
    print(f"   ✅ SUCCESS: All data indices are covered by vocabulary!")
    if data_indices == vocab_indices:
        print(f"   ✅ PERFECT: Vocabulary exactly matches data (no extra types)")
    else:
        extra_in_vocab = vocab_indices - data_indices
        print(f"   ⚠️  WARNING: Vocabulary has extra types not in data: {extra_in_vocab}")
else:
    missing = data_indices - vocab_indices
    print(f"   ❌ ERROR: Data has indices not in vocabulary: {missing}")
    print(f"   This will cause NaN errors during training!")

print(f"\n4. Model Implications:")
print(f"   - Embedding layer size: {len(vocab['object_idx_to_name'])}")
print(f"   - Refinement net output channels: {len(vocab['object_idx_to_name'])}")
print(f"   - Max index in data: {max(unique_data_indices)}")
print(f"   - Embedding must support indices: 0-{max(unique_data_indices)}")

# Check if sparse indexing is safe
max_index = max(unique_data_indices)
vocab_size = len(vocab['object_idx_to_name'])
if max_index >= vocab_size:
    print(f"\n   ⚠️  WARNING: Sparse indexing detected!")
    print(f"   - Max data index ({max_index}) >= vocab size ({vocab_size})")
    print(f"   - Embedding layer will have unused indices: {set(range(max_index+1)) - data_indices}")
    print(f"   - This is OK but inefficient. Consider remapping data to 0-{vocab_size-1}")
else:
    print(f"\n   ✅ Contiguous indexing: Max index ({max_index}) < vocab size ({vocab_size})")

print("\n" + "=" * 80)
