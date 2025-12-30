"""
Scan full dataset to identify all room types present
"""
import scipy.io as sio
import numpy as np
from collections import Counter

print("Scanning training data...")
data_train = sio.loadmat('./data/data_train.mat', squeeze_me=True, struct_as_record=False)['data']

print("Scanning validation data...")
try:
    data_valid = sio.loadmat('./data/data_valid.mat', squeeze_me=True, struct_as_record=False)['data']
except:
    print("  No validation data found")
    data_valid = []

print("Scanning test data...")
try:
    data_test = sio.loadmat('./data/data_test.mat', squeeze_me=True, struct_as_record=False)['data']
except:
    print("  No test data found")
    data_test = []

def scan_room_types(data, name):
    print(f"\n{'='*60}")
    print(f"Scanning {name} set: {len(data)} samples")
    print(f"{'='*60}")
    
    all_room_types = []
    skipped = 0
    for item in data:
        # Check if item has 'box' attribute
        if not hasattr(item, 'box'):
            skipped += 1
            continue
        room_types = item.box[:, 4].astype(int).tolist()
        all_room_types.extend(room_types)
    
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} samples without 'box' attribute")
    
    if len(all_room_types) == 0:
        print(f"❌ No valid room data found in {name} set")
        return Counter()
    
    # Count occurrences
    type_counts = Counter(all_room_types)
    
    print(f"\nUnique room types found: {sorted(type_counts.keys())}")
    print(f"Min room type: {min(type_counts.keys())}")
    print(f"Max room type: {max(type_counts.keys())}")
    print(f"Total rooms: {len(all_room_types)}")
    
    print(f"\nRoom type distribution:")
    for room_type in sorted(type_counts.keys()):
        count = type_counts[room_type]
        percentage = (count / len(all_room_types)) * 100
        print(f"  Type {room_type:2d}: {count:6d} rooms ({percentage:5.2f}%)")
    
    return type_counts

# Scan all datasets
train_counts = scan_room_types(data_train, "Training")

if len(data_valid) > 0:
    valid_counts = scan_room_types(data_valid, "Validation")

if len(data_test) > 0:
    test_counts = scan_room_types(data_test, "Test")

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")

all_types = set(train_counts.keys())
if len(data_valid) > 0:
    all_types.update(valid_counts.keys())
if len(data_test) > 0:
    all_types.update(test_counts.keys())

print(f"\nAll room types across all datasets: {sorted(all_types)}")
print(f"Max room type ID: {max(all_types)}")
print(f"\nVocabulary size needed: {max(all_types) + 1}")

# Check for missing types
expected_types = set(range(max(all_types) + 1))
missing_types = expected_types - all_types
if missing_types:
    print(f"\n⚠️  Missing room type IDs: {sorted(missing_types)}")
    print("   (These types are not present in the dataset)")
else:
    print(f"\n✅ All room type IDs from 0 to {max(all_types)} are present")

# Check if types 15, 16, 17 exist
print(f"\n{'='*60}")
print("ROOM TYPES 15-17 CHECK")
print(f"{'='*60}")
for t in [15, 16, 17]:
    if t in all_types:
        total = train_counts.get(t, 0) + valid_counts.get(t, 0) if len(data_valid) > 0 else train_counts.get(t, 0)
        print(f"✅ Type {t}: EXISTS ({total} instances)")
    else:
        print(f"❌ Type {t}: NOT FOUND in dataset")

print("\n" + "="*60)
