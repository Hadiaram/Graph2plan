"""
Quick analysis: Count how many rooms are complex polygons vs rectangles
"""

import scipy.io as sio
import numpy as np
from pathlib import Path

def analyze_room_complexity(sample, sample_idx):
    """
    Check if rooms are polygons or rectangles
    Returns: (num_polygons, num_rectangles, details)
    """
    rBoundary = sample.rBoundary if hasattr(sample, 'rBoundary') else None
    boxes = sample.box if hasattr(sample, 'box') else None
    
    if rBoundary is None or boxes is None:
        return 0, 0, []
    
    num_polygons = 0
    num_rectangles = 0
    details = []
    
    for i, (rb, box) in enumerate(zip(rBoundary, boxes)):
        if rb is None or not isinstance(rb, np.ndarray):
            continue
        
        # Get number of vertices
        if rb.ndim == 2 and rb.shape[1] >= 2:
            num_vertices = rb.shape[0]
        elif rb.ndim == 1:
            num_vertices = len(rb) // 2
        else:
            continue
        
        # Get room type
        room_type = int(box[0]) if len(box) > 0 else -1
        
        # Classify as polygon or rectangle
        if num_vertices == 4:
            num_rectangles += 1
            shape_type = "RECT"
        else:
            num_polygons += 1
            shape_type = "POLY"
        
        details.append({
            'room_idx': i,
            'room_type': room_type,
            'vertices': num_vertices,
            'shape': shape_type
        })
    
    return num_polygons, num_rectangles, details


def main():
    print("\n" + "="*80)
    print("ROOM SHAPE COMPLEXITY ANALYSIS")
    print("="*80 + "\n")
    
    train_file = Path('../Network/data/data_train.mat')
    
    if not train_file.exists():
        print(f"ERROR: {train_file} not found")
        return
    
    print(f"Loading: {train_file}")
    data = sio.loadmat(train_file, squeeze_me=True, struct_as_record=False)
    samples = data['data']
    
    print(f"Total samples: {len(samples)}\n")
    
    # Analyze first 20 samples
    total_polygons = 0
    total_rectangles = 0
    
    num_samples = min(20, len(samples))
    
    for i in range(num_samples):
        num_poly, num_rect, details = analyze_room_complexity(samples[i], i)
        total_polygons += num_poly
        total_rectangles += num_rect
        
        print(f"Sample {i:3d}: {num_poly} polygons, {num_rect} rectangles")
        
        # Show details for first 5 samples
        if i < 5:
            for room in details:
                print(f"  Room {room['room_idx']}: {room['vertices']:2d} vertices - {room['shape']}")
        
        if i == 4:
            print()
    
    print(f"\n{'='*80}")
    print(f"SUMMARY (first {num_samples} samples):")
    print(f"{'='*80}")
    print(f"Total complex polygons (>4 vertices): {total_polygons}")
    print(f"Total rectangles (4 vertices):        {total_rectangles}")
    print(f"Percentage of complex shapes:          {100*total_polygons/(total_polygons+total_rectangles):.1f}%")
    print(f"{'='*80}")
    
    print(f"\n{'='*80}")
    print("CONCLUSION:")
    print(f"{'='*80}")
    if total_polygons > 0:
        print("✓ The processed data DOES preserve complex polygon shapes!")
        print("✓ Rooms are stored in 'rBoundary' field with their actual vertices.")
        print("✓ The 'box' field contains bounding rectangles for computational efficiency.")
        print("\nIMPLICATION:")
        print("  - Graph2Plan CAN generate complex room shapes (data is available)")
        print("  - The rectangular-only output is a VISUALIZATION or MODEL limitation")
        print("  - Check the model's refinement_net output and layout generation")
    else:
        print("✗ All rooms are rectangularized (4 vertices only)")
        print("✗ Complex shapes are NOT preserved in the processed data")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
