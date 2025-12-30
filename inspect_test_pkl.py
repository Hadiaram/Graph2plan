"""
Quick diagnostic script to inspect the structure of data_test_converted.pkl
"""
import pickle
import numpy as np

TEST_PKL = r"C:\Users\hmbashir\AI Training\Graph2Plan\Interface\static\Data\data_test_converted.pkl"

print("Loading test PKL file...")
with open(TEST_PKL, 'rb') as f:
    data = pickle.load(f)

print(f"\n=== Top-level structure ===")
print(f"Type: {type(data)}")

if isinstance(data, dict):
    print(f"Keys: {list(data.keys())}")
    print(f"\nInspecting each key:")
    for key, val in data.items():
        print(f"  {key}: type={type(val)}, ", end="")
        if isinstance(val, np.ndarray):
            print(f"shape={val.shape}, dtype={val.dtype}")
        elif isinstance(val, (list, tuple)):
            print(f"len={len(val)}")
        else:
            print(f"value={val}")
    
    # Get the floor plans
    if 'data' in data:
        floor_plans = data['data']
    else:
        floor_plans = data
else:
    floor_plans = data

# Handle numpy array squeezing
if not isinstance(floor_plans, (list, np.ndarray)):
    floor_plans = [floor_plans]
elif isinstance(floor_plans, np.ndarray):
    if floor_plans.ndim == 0:
        floor_plans = [floor_plans.item()]
    elif floor_plans.ndim == 1:
        floor_plans = list(floor_plans)
    else:
        floor_plans = list(floor_plans.flat)

print(f"\n=== Floor plans ===")
print(f"Number of floor plans: {len(floor_plans)}")

if len(floor_plans) > 0:
    print(f"\n=== First floor plan structure ===")
    fp = floor_plans[0]
    print(f"Type: {type(fp)}")
    
    # Try to list all attributes
    if hasattr(fp, '__dict__'):
        print(f"Attributes: {list(fp.__dict__.keys())}")
    elif hasattr(fp, '_fieldnames'):
        print(f"Fields: {fp._fieldnames}")
    elif hasattr(fp, 'dtype'):
        print(f"Dtype fields: {fp.dtype.names}")
    elif isinstance(fp, dict):
        print(f"Keys: {list(fp.keys())}")
    
    # Try to access common attributes
    print(f"\nTrying to access common attributes:")
    for attr in ['box', 'edge', 'name', 'boundary', 'rType', 'rBoundary', 'gtBox', 'gtBoxNew', 'rEdge', 'order']:
        try:
            if isinstance(fp, dict):
                val = fp.get(attr)
            else:
                val = getattr(fp, attr, None)
            
            if val is not None:
                if isinstance(val, np.ndarray):
                    print(f"  {attr}: array, shape={val.shape}, dtype={val.dtype}")
                elif isinstance(val, (list, tuple)):
                    print(f"  {attr}: {type(val).__name__}, len={len(val)}")
                else:
                    print(f"  {attr}: {type(val).__name__}")
            else:
                print(f"  {attr}: None or missing")
        except Exception as e:
            print(f"  {attr}: Error - {e}")
    
    # Check a few more floor plans
    print(f"\n=== Checking first 5 floor plans for 'box' attribute ===")
    for i in range(min(5, len(floor_plans))):
        fp = floor_plans[i]
        try:
            if isinstance(fp, dict):
                has_box = 'box' in fp
                box_val = fp.get('box') if has_box else None
            else:
                has_box = hasattr(fp, 'box')
                box_val = getattr(fp, 'box', None) if has_box else None
            
            print(f"  Floor plan {i}: has_box={has_box}, box is None={box_val is None}")
        except Exception as e:
            print(f"  Floor plan {i}: Error - {e}")

print("\n=== Done ===")
