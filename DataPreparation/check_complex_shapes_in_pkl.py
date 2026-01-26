"""
Check if Interface pkl files contain complex polygon shapes (rBoundary field)
vs just rectangular bounding boxes (box field).

This will determine if the rectangular-only output is a:
1. Data limitation (dataset only has rectangles)
2. Model limitation (model ignores polygon data)
"""

import pickle
import numpy as np
from pathlib import Path

def analyze_room_shapes_in_pkl(pkl_path):
    """
    Analyze whether the pkl file contains complex polygon shapes
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {pkl_path}")
    print(f"{'='*80}\n")
    
    with open(pkl_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    data = data_dict['data']
    print(f"Total samples: {len(data)}")
    
    # Check first sample structure
    first = data[0]
    print(f"\nFirst sample attributes: {[a for a in dir(first) if not a.startswith('_')]}")
    
    # Check if rBoundary exists
    has_rboundary = hasattr(first, 'rBoundary')
    print(f"\nHas 'rBoundary' field: {has_rboundary}")
    
    if has_rboundary:
        print("\n✓ The data DOES contain polygon boundaries (rBoundary field)")
        print("  This field stores the actual room shapes as polygons, not just rectangles\n")
        
        # Analyze shape complexity
        total_rooms = 0
        rectangular_rooms = 0
        complex_rooms = 0
        vertex_counts = []
        
        # Sample 100 floor plans
        sample_size = min(100, len(data))
        print(f"Analyzing room shapes in {sample_size} samples...\n")
        
        for i in range(sample_size):
            sample = data[i]
            rBoundary = sample.rBoundary
            
            if rBoundary is not None:
                for room_boundary in rBoundary:
                    if room_boundary is not None and len(room_boundary) > 0:
                        total_rooms += 1
                        
                        # Count vertices
                        if isinstance(room_boundary, np.ndarray):
                            num_vertices = len(room_boundary)
                        else:
                            num_vertices = 4  # Default assumption
                        
                        vertex_counts.append(num_vertices)
                        
                        if num_vertices == 4:
                            rectangular_rooms += 1
                        else:
                            complex_rooms += 1
                            
                            # Show example complex shapes
                            if complex_rooms <= 5:
                                print(f"  Example complex room {complex_rooms}: {num_vertices} vertices")
        
        print(f"\n{'='*80}")
        print("ROOM SHAPE STATISTICS:")
        print(f"{'='*80}")
        print(f"Total rooms analyzed: {total_rooms}")
        print(f"Rectangular (4 vertices): {rectangular_rooms} ({100*rectangular_rooms/total_rooms:.1f}%)")
        print(f"Complex shapes (>4 vertices): {complex_rooms} ({100*complex_rooms/total_rooms:.1f}%)")
        
        if vertex_counts:
            print(f"\nVertex count range: {min(vertex_counts)} to {max(vertex_counts)}")
            print(f"Average vertices per room: {np.mean(vertex_counts):.1f}")
        
        print(f"\n{'='*80}")
        print("CONCLUSION:")
        print(f"{'='*80}")
        if complex_rooms > 0:
            print("✓ The dataset CONTAINS COMPLEX POLYGON SHAPES")
            print("✗ The rectangular-only output is a MODEL LIMITATION")
            print("\nThe model uses 'box' field (rectangles) and IGNORES 'rBoundary' (polygons)")
            print("This is by design - Graph2Plan generates rectangular bounding boxes only.")
        else:
            print("✗ The dataset contains only rectangular rooms")
            print("✓ The rectangular output matches the data")
        print(f"{'='*80}\n")
        
    else:
        print("\n✗ No 'rBoundary' field found")
        print("  The data only contains bounding boxes (box field)")
        print("  → Rectangular output is expected (data limitation)\n")
    
    # Show the difference between box and rBoundary
    print("\n" + "="*80)
    print("DATA STRUCTURE COMPARISON:")
    print("="*80)
    
    sample = data[0]
    
    print("\n1. 'box' field (what the model USES):")
    if hasattr(sample, 'box'):
        box = sample.box
        print(f"   Shape: {box.shape}")
        print(f"   Format: [room_type, y0, x0, y1, x1] - RECTANGULAR BOUNDING BOX")
        print(f"   Example: {box[0]}")
    
    print("\n2. 'rBoundary' field (what the model IGNORES):")
    if hasattr(sample, 'rBoundary'):
        rBoundary = sample.rBoundary
        print(f"   Type: {type(rBoundary)}")
        print(f"   Number of rooms: {len(rBoundary)}")
        if len(rBoundary) > 0 and rBoundary[0] is not None:
            print(f"   Format: Array of (x,y) polygon vertices - COMPLEX SHAPES")
            print(f"   Example room 0: {rBoundary[0].shape if hasattr(rBoundary[0], 'shape') else 'N/A'}")
            if hasattr(rBoundary[0], 'shape') and len(rBoundary[0]) > 0:
                print(f"   First 3 vertices: {rBoundary[0][:3]}")
    
    print("="*80 + "\n")


def main():
    print("\n" + "="*80)
    print("CHECKING IF DATASET CONTAINS COMPLEX ROOM SHAPES")
    print("="*80)
    
    pkl_path = Path('../Interface/static/Data/data_train_converted.pkl')
    
    if not pkl_path.exists():
        pkl_path = Path('../../Interface/static/Data/data_train_converted.pkl')
    
    if not pkl_path.exists():
        print("ERROR: Cannot find data_train_converted.pkl")
        return
    
    analyze_room_shapes_in_pkl(pkl_path)


if __name__ == '__main__':
    main()
