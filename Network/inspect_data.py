"""
Diagnostic script to inspect the structure of data_train.mat
"""
import scipy.io as sio
import numpy as np

DATA_PATH = './data/data_train.mat'

print("Loading data...")
data = sio.loadmat(DATA_PATH, squeeze_me=True, struct_as_record=False)

print(f"\n=== Top-level keys ===")
print(f"Keys: {list(data.keys())}")

if 'data' in data:
    data_array = data['data']
    print(f"\n=== Data array ===")
    print(f"Type: {type(data_array)}")
    print(f"Length: {len(data_array)}")
    
    if len(data_array) > 0:
        print(f"\n=== First data item ===")
        item = data_array[0]
        print(f"Type: {type(item)}")
        
        # Check if it has _fieldnames
        if hasattr(item, '_fieldnames'):
            print(f"Field names: {item._fieldnames}")
            
            # Print each field
            for field in item._fieldnames:
                try:
                    val = getattr(item, field)
                    if isinstance(val, np.ndarray):
                        print(f"  {field}: array, shape={val.shape}, dtype={val.dtype}")
                    else:
                        print(f"  {field}: {type(val).__name__}")
                except Exception as e:
                    print(f"  {field}: Error - {e}")
        
        # Check for common expected attributes
        print(f"\n=== Checking for expected attributes ===")
        expected = ['name', 'boundary', 'order', 'rType', 'rBoundary', 'gtBox', 'gtBoxNew', 'rEdge']
        for attr in expected:
            has_attr = hasattr(item, attr)
            print(f"  {attr}: {has_attr}")
            if has_attr:
                val = getattr(item, attr)
                if isinstance(val, np.ndarray):
                    print(f"    -> shape={val.shape}, dtype={val.dtype}")

print("\n=== Done ===")
