"""
Check data.mat for NaN, Inf, or extreme values that could cause training to fail.
"""
import scipy.io as sio
import numpy as np

def check_data_integrity(data_path='./data.mat'):
    print(f"Loading {data_path}...")
    data = sio.loadmat(data_path)
    
    print("\n=== Data Keys ===")
    for key in data.keys():
        if not key.startswith('__'):
            print(f"  {key}: {type(data[key])}")
    
    issues_found = False
    
    # Check all arrays in the data
    for key in data.keys():
        if key.startswith('__'):
            continue
            
        value = data[key]
        
        # Skip if not array-like
        if not isinstance(value, np.ndarray):
            continue
        
        # Check for NaN
        if np.issubdtype(value.dtype, np.floating):
            nan_count = np.isnan(value).sum()
            inf_count = np.isinf(value).sum()
            
            if nan_count > 0:
                print(f"\n⚠️  WARNING: {key} contains {nan_count} NaN values!")
                issues_found = True
            
            if inf_count > 0:
                print(f"\n⚠️  WARNING: {key} contains {inf_count} Inf values!")
                issues_found = True
            
            # Check for extreme values
            if value.size > 0:
                max_val = np.abs(value).max()
                if max_val > 1e6:
                    print(f"\n⚠️  WARNING: {key} has extreme values (max={max_val:.2e})!")
                    issues_found = True
                    
            print(f"✓ {key}: shape={value.shape}, dtype={value.dtype}, "
                  f"range=[{value.min():.4f}, {value.max():.4f}]")
    
    if not issues_found:
        print("\n✅ No NaN/Inf/extreme values found in dataset!")
    else:
        print("\n❌ Dataset has integrity issues that may cause NaN during training!")
    
    return not issues_found

if __name__ == '__main__':
    check_data_integrity()
