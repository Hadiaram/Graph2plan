"""
Check validation and test data for unmapped indices
"""
import scipy.io as sio
import numpy as np
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from model.utils import get_vocab

vocab = get_vocab()
valid_data_indices = set(vocab['data_idx_to_model_idx'].keys())

print("=" * 70)
print("CHECKING VALIDATION AND TEST DATA")
print("=" * 70)

for data_file in ['data_train.mat', 'data_valid.mat', 'data_test.mat']:
    print(f"\n{data_file}:")
    try:
        data = sio.loadmat(f'./data/{data_file}', squeeze_me=True, struct_as_record=False)
        
        all_indices = set()
        problem_indices = set()
        
        for sample in data['data']:
            if hasattr(sample, 'rType'):
                room_types = sample.rType if isinstance(sample.rType, np.ndarray) else [sample.rType]
            elif hasattr(sample, 'box'):
                room_types = sample.box[:, 4].astype(int)
            else:
                continue
            
            for idx in room_types:
                all_indices.add(int(idx))
                if int(idx) not in valid_data_indices:
                    problem_indices.add(int(idx))
        
        print(f"  All indices: {sorted(all_indices)}")
        
        if problem_indices:
            print(f"  ❌ UNMAPPED INDICES: {sorted(problem_indices)}")
        else:
            print(f"  ✅ All indices mapped")
            
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 70)
