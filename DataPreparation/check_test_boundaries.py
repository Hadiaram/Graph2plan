"""Check if test data boundaries are all the same"""
import pickle
import numpy as np

# Load test data
print("Loading test data...")
with open('../Interface/static/Data/data_test_converted.pkl', 'rb') as f:
    test_data = pickle.load(f)

print('Keys in PKL:', list(test_data.keys()))
print('Number of test floor plans:', len(test_data['data']))
print('Number of test names:', len(test_data['testNameList']))

# Check first 10 floor plan names and their boundary shapes
print('\nFirst 10 test floor plans:')
for i in range(min(10, len(test_data['data']))):
    item = test_data['data'][i]
    name = item.name if hasattr(item, 'name') else 'unknown'
    boundary_shape = item.boundary.shape if hasattr(item, 'boundary') else 'no boundary'
    rBoundary_len = len(item.rBoundary) if hasattr(item, 'rBoundary') else 'no rBoundary'
    print(f'  [{i}] name={name}, boundary shape={boundary_shape}, rBoundary rooms={rBoundary_len}')

# Check if all boundaries are the same
print('\nChecking if all boundaries are identical...')
if len(test_data['data']) > 1:
    first_boundary = test_data['data'][0].boundary
    differences = []
    for i in range(1, len(test_data['data'])):
        if not np.array_equal(first_boundary, test_data['data'][i].boundary):
            differences.append(i)
            if len(differences) <= 3:
                print(f'  Floor plan {i} (name={test_data["data"][i].name}) has DIFFERENT boundary')

    if len(differences) == 0:
        print('  ⚠️ WARNING: ALL boundaries are IDENTICAL! This is the bug!')
        print(f'  All {len(test_data["data"])} floor plans have the same boundary as floor plan 0')
    else:
        print(f'  ✓ Good: {len(differences)} floor plans have different boundaries')
else:
    print('  Only 1 floor plan in test data')

# Check testNameList
print('\nTest name list (first 10):')
print(test_data['testNameList'][:10])
