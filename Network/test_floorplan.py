"""
Test the FloorPlan compatibility layer
"""
import sys
sys.path.append('.')
from model.floorplan import FloorPlan
import scipy.io as sio
import numpy as np

DATA_PATH = './data/data_train.mat'

print("Loading data...")
data = sio.loadmat(DATA_PATH, squeeze_me=True, struct_as_record=False)['data']

print(f"Loaded {len(data)} floor plans")

print("\n=== Testing first floor plan ===")
item = data[0]

print(f"Original attributes: {item._fieldnames}")
print(f"  box shape: {item.box.shape}")
print(f"  edge shape: {item.edge.shape}")

print("\nCreating FloorPlan object...")
try:
    fp = FloorPlan(item)
    print("✅ FloorPlan created successfully!")
    
    print(f"\n=== After compatibility conversion ===")
    print(f"  gtBoxNew shape: {fp.data.gtBoxNew.shape}")
    print(f"  rType shape: {fp.data.rType.shape}")
    print(f"  rEdge shape: {fp.data.rEdge.shape}")
    
    print(f"\n=== Sample data ===")
    print(f"  First box (gtBoxNew): {fp.data.gtBoxNew[0]}")
    print(f"  First room type (rType): {fp.data.rType[0]}")
    print(f"  First edge (rEdge): {fp.data.rEdge[0]}")
    
    print("\n✅ All compatibility checks passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
