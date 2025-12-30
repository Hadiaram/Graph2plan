"""
Quick check to see what room types exist in training data
and predict what errors we'll hit next
"""
import scipy.io as sio
import numpy as np

print("="*60)
print("CHECKING DATA FOR POTENTIAL ERRORS")
print("="*60)

# Load training data
print("\nLoading data/data_train.mat...")
data = sio.loadmat('./data/data_train.mat', squeeze_me=True, struct_as_record=False)['data']

print(f"Found {len(data)} floor plans")

# Check room types
print("\n1. Checking room types...")
all_room_types = []
max_boundary_x = 0
max_boundary_y = 0

for item in data[:1000]:  # Check first 1000 samples
    # Room types
    room_types = item.box[:, 4]
    all_room_types.extend(room_types.tolist())
    
    # Boundary coordinates
    boundary = item.boundary[:, :2]
    max_boundary_x = max(max_boundary_x, np.max(boundary[:, 0]))
    max_boundary_y = max(max_boundary_y, np.max(boundary[:, 1]))

unique_types = sorted(set(all_room_types))
min_type = min(all_room_types)
max_type = max(all_room_types)

print(f"   Unique room types: {unique_types}")
print(f"   Min room type: {min_type}")
print(f"   Max room type: {max_type}")

# Check boundary coordinates
print(f"\n2. Checking boundary coordinates...")
print(f"   Max boundary X: {max_boundary_x}")
print(f"   Max boundary Y: {max_boundary_y}")

# Predictions
print("\n" + "="*60)
print("PREDICTED ERRORS")
print("="*60)

error_count = 3

# Error #3: Embedding index out of range
if max_type > 14:
    print(f"\n❌ ERROR #{error_count} WILL OCCUR: Embedding index out of range")
    print(f"   Reason: Data has room type {int(max_type)}, but model only supports 0-14")
    print(f"   Message: 'IndexError: index {int(max_type)} is out of bounds for axis 0 with size 15'")
    print(f"   Location: model/model.py when processing embeddings")
    print(f"   Fix: Change model embeddings from 15 to {int(max_type)+1} types")
    error_count += 1
else:
    print(f"\n✅ No embedding error expected - max room type is {int(max_type)}")

# Error #4: Loss weight mismatch
if max_type > 14:
    print(f"\n❌ ERROR #{error_count} WILL OCCUR: Loss weight size mismatch")
    print(f"   Reason: If you fix embeddings to size {int(max_type)+1}, loss expects same size")
    print(f"   Message: 'RuntimeError: size mismatch'")
    print(f"   Location: train.py loss computation")
    print(f"   Fix: Change torch.ones(15) to torch.ones({int(max_type)+1})")
    error_count += 1

# Error #5: Boundary clipping
if max_boundary_x >= 256 or max_boundary_y >= 256:
    print(f"\n⚠️  ERROR #{error_count} MIGHT OCCUR: Boundary index out of range")
    print(f"   Reason: Boundary coords reach {max_boundary_x}, {max_boundary_y}")
    print(f"   Message: 'IndexError: index 256 is out of bounds'")
    print(f"   Location: floorplan.py adjust_graph() method")
    print(f"   Fix: Add clipping to [0, 255] range")
    error_count += 1
else:
    print(f"\n✅ Boundary coordinates look safe (max {max_boundary_x}, {max_boundary_y})")

# Summary
print("\n" + "="*60)
print("DECISION NEEDED")
print("="*60)

if max_type > 14:
    print(f"\n🔴 Your data has room types 0-{int(max_type)} ({int(max_type)+1} types total)")
    print(f"   Current model only supports 15 types (0-14)")
    print(f"\n   You MUST choose:")
    print(f"\n   Option A: Train with {int(max_type)+1} types")
    print(f"      - Follow TRAINING_GUIDE_18_ROOMS.md")
    print(f"      - Update model embeddings: 15 → {int(max_type)+1}")
    print(f"      - Update loss weights: 15 → {int(max_type)+1}")
    print(f"      - Add vocab entries for types 15-17")
    print(f"\n   Option B: Preprocess data to only use 15 types")
    print(f"      - Map/remove room types 15-{int(max_type)}")
    print(f"      - Regenerate data files")
    print(f"      - Current code will work as-is")
    print(f"\n   ⚠️  RECOMMENDED: Option A (already have the data)")
else:
    print(f"\n🟢 Your data only has room types 0-{int(max_type)} (15 types)")
    print(f"   Current model configuration (15 types) is CORRECT")
    print(f"   Training should work after coordinate bug fix!")

print("\n" + "="*60)
