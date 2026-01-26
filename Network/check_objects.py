"""
Check what room types are actually in the current data.mat
"""
import scipy.io as sio
import numpy as np

data = sio.loadmat('./data.mat')
print("Loading data.mat...")

# Extract all object types from the dataset
all_objs = []
for sample in data['data'][0]:
    objs = sample[3][0]  # Index 3 is 'objs'
    all_objs.extend(objs)

unique_objs = np.unique(all_objs)
print(f"\nUnique object indices in data: {unique_objs}")
print(f"Number of unique objects: {len(unique_objs)}")
print(f"Range: {unique_objs.min()} to {unique_objs.max()}")
