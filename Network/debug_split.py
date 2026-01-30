import scipy.io as sio
import numpy as np

print("Loading data.mat...")
data = sio.loadmat('data.mat', squeeze_me=True, struct_as_record=False)['data']
print(f"Total samples loaded: {len(data) if hasattr(data, '__len__') else 1}")
print(f"Data type: {type(data)}")

len_train, len_valid = int(len(data)*0.70), int(len(data)*0.15)
print(f"\nSplit sizes:")
print(f"  len_train: {len_train} (70%)")
print(f"  len_valid: {len_valid} (15%)")
print(f"  len_test: {len(data) - len_train - len_valid} (15%)")

data_train = data[:len_train]
data_valid = data[len_train:len_train+len_valid]
data_test = data[len_train+len_valid:]

print(f"\nActual split arrays:")
print(f"  data_train type: {type(data_train)}, size: {len(data_train) if hasattr(data_train, '__len__') else 'scalar'}")
print(f"  data_valid type: {type(data_valid)}, size: {len(data_valid) if hasattr(data_valid, '__len__') else 'scalar'}")
print(f"  data_test type: {type(data_test)}, size: {len(data_test) if hasattr(data_test, '__len__') else 'scalar'}")

# Check if validation array is actually empty
if hasattr(data_valid, '__len__'):
    if len(data_valid) == 0:
        print("\n❌ WARNING: data_valid is EMPTY!")
    else:
        print(f"\n✓ data_valid has {len(data_valid)} samples")
        # Check first validation sample
        print(f"\nFirst validation sample:")
        print(f"  Type: {type(data_valid[0])}")
        if hasattr(data_valid[0], 'name'):
            print(f"  Name: {data_valid[0].name}")
        if hasattr(data_valid[0], 'rType'):
            print(f"  rType: {data_valid[0].rType}")
else:
    print("\n❌ WARNING: data_valid is not an array!")
    print(f"  It's a: {type(data_valid)}")

# Try to save and see what happens
print("\n" + "="*60)
print("Attempting to save files...")
print("="*60)

try:
    print("Saving data_train.mat...")
    sio.savemat('./data/debug_data_train.mat', {'data': data_train})
    print("  ✓ Success")
except Exception as e:
    print(f"  ❌ Failed: {e}")

try:
    print("Saving data_valid.mat...")
    sio.savemat('./data/debug_data_valid.mat', {'data': data_valid})
    print("  ✓ Success")
except Exception as e:
    print(f"  ❌ Failed: {e}")

try:
    print("Saving data_test.mat...")
    sio.savemat('./data/debug_data_test.mat', {'data': data_test})
    print("  ✓ Success")
except Exception as e:
    print(f"  ❌ Failed: {e}")

import os
print("\n" + "="*60)
print("Checking saved file sizes:")
print("="*60)
for fname in ['debug_data_train.mat', 'debug_data_valid.mat', 'debug_data_test.mat']:
    fpath = f'./data/{fname}'
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        print(f"{fname}: {size:,} bytes")
    else:
        print(f"{fname}: NOT CREATED")
