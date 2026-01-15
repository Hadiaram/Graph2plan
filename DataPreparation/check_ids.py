import scipy.io as sio
import random

# Load data.mat
data = sio.loadmat('data/data.mat')
items = data['data'][0]

print(f"Total items in data.mat: {len(items)}")

# Extract all IDs (filter out empty strings)
all_ids = []
empty_count = 0
for i in range(len(items)):
    name = items[i]['name'][0]
    if isinstance(name, bytes):
        name = name.decode('utf-8')
    elif hasattr(name, 'item'):
        name = str(name.item())
    else:
        name = str(name)
    
    name = name.strip()
    if name:  # Only add non-empty IDs
        all_ids.append(name)
    else:
        empty_count += 1
        print(f"Warning: Empty ID at index {i}")

if empty_count > 0:
    print(f"\nFiltered out {empty_count} empty IDs")
print(f"Valid IDs: {len(all_ids)}")

print(f"\nFirst 20 IDs:")
for i in range(min(20, len(all_ids))):
    print(f"  {i}: {all_ids[i]}")

print(f"\nLast 10 IDs:")
for i in range(max(0, len(all_ids)-10), len(all_ids)):
    print(f"  {i}: {all_ids[i]}")

# Use all samples and split 80/20
random.seed(42)
selected_ids = all_ids.copy()
random.shuffle(selected_ids)

train_size = int(len(selected_ids) * 0.8)
train_ids = selected_ids[:train_size]
test_ids = selected_ids[train_size:]

# Write train.txt (no trailing newline)
with open('data/train.txt', 'w') as f:
    for i, id in enumerate(train_ids):
        if i == len(train_ids) - 1:
            f.write(f"{id}")  # Last line without newline
        else:
            f.write(f"{id}\n")

# Write test.txt (no trailing newline)
with open('data/test.txt', 'w') as f:
    for i, id in enumerate(test_ids):
        if i == len(test_ids) - 1:
            f.write(f"{id}")  # Last line without newline
        else:
            f.write(f"{id}\n")

print(f"\n" + "="*60)
print("✅ Created train.txt and test.txt")
print("="*60)
print(f"Total samples: {len(selected_ids)}")
print(f"Train: {len(train_ids)} samples → data/train.txt")
print(f"Test:  {len(test_ids)} samples → data/test.txt")
print(f"\nNext: python 1.tf_train.py")
