"""
Visual demonstration of the key finding:
- Training data HAS complex shapes in rBoundary (from ResPlan)
- Model ignores them and only uses rectangular boxes
- Post-processing creates NEW complex shapes from rectangle intersections
- These post-processed shapes are NOT the same as the original training shapes
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

ROOM_COLORS = {0: '#EE4D4D', 1: '#C67171', 2: '#FFD274', 3: '#BEBEBE', 4: '#D11A2D'}
ROOM_NAMES = {0: 'Living', 1: 'Master', 2: 'Kitchen', 3: 'Bath', 4: 'Door'}


def visualize_data_flow(sample, sample_idx, name, output_path):
    """
    Show the complete data flow:
    1. Original complex shapes (rBoundary from training data)
    2. Rectangular boxes (what model uses/generates)
    3. Post-processed shapes (rBoundary derived from rectangles)
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(f'Sample {sample_idx}: Data Flow Analysis\n'
                 f'Showing why rectangles dominate the pipeline',
                 fontsize=15, fontweight='bold')
    
    boundary = sample.boundary
    boxes = sample.box
    rBoundary = sample.rBoundary
    
    boundary_coords = boundary[:, :2]
    
    # ===== PANEL 1: Original Training Data (Complex Shapes) =====
    ax1 = axes[0]
    ax1.set_title('TRAINING DATA STORAGE\n'
                  'Original complex shapes from ResPlan\n'
                  '(Stored in rBoundary field)',
                  fontsize=11, fontweight='bold', color='green')
    
    # Boundary
    boundary_polygon = patches.Polygon(
        boundary_coords, fill=True, facecolor='white',
        edgecolor='black', linewidth=3, alpha=0.3
    )
    ax1.add_patch(boundary_polygon)
    
    # Plot original complex polygons
    complex_count = 0
    total_vertices = 0
    for i, (room_poly, box) in enumerate(zip(rBoundary, boxes)):
        if room_poly is not None and len(room_poly) > 0:
            room_type = int(box[0])
            color = ROOM_COLORS.get(room_type, '#CCCCCC')
            
            num_vertices = len(room_poly)
            total_vertices += num_vertices
            if num_vertices > 4:
                complex_count += 1
            
            poly_patch = patches.Polygon(
                room_poly, fill=True, facecolor=color,
                edgecolor='darkgreen', linewidth=2.5, alpha=0.7
            )
            ax1.add_patch(poly_patch)
    
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.autoscale()
    
    info = f"✓ Complex shapes preserved\n"
    info += f"✓ {complex_count}/{len(boxes)} rooms >4 vertices\n"
    info += f"✓ Avg {total_vertices/len(boxes):.1f} vertices/room\n"
    info += f"✗ MODEL IGNORES THIS FIELD"
    ax1.text(0.02, 0.02, info,
            transform=ax1.transAxes, fontsize=10, color='darkgreen',
            fontweight='bold', verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
    
    # ===== PANEL 2: What Model Uses (Rectangles) =====
    ax2 = axes[1]
    ax2.set_title('MODEL TRAINING & INFERENCE\n'
                  'Only uses rectangular bounding boxes\n'
                  '(From box field)',
                  fontsize=11, fontweight='bold', color='blue')
    
    # Boundary
    boundary_polygon = patches.Polygon(
        boundary_coords, fill=True, facecolor='white',
        edgecolor='black', linewidth=3, alpha=0.3
    )
    ax2.add_patch(boundary_polygon)
    
    # Plot rectangular boxes
    for i, box in enumerate(boxes):
        room_type = int(box[0])
        y0, x0, y1, x1 = box[1:5]
        
        color = ROOM_COLORS.get(room_type, '#CCCCCC')
        room_name = ROOM_NAMES.get(room_type, f'Type{room_type}')
        
        x_min, x_max = min(x0, x1), max(x0, x1)
        y_min, y_max = min(y0, y1), max(y0, y1)
        width = x_max - x_min
        height = y_max - y_min
        
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2.5, edgecolor='darkblue',
            facecolor=color, alpha=0.7
        )
        ax2.add_patch(rect)
        
        # Add corner markers to emphasize rectangles
        for corner_x in [x_min, x_max]:
            for corner_y in [y_min, y_max]:
                ax2.plot(corner_x, corner_y, 'bo', markersize=6)
    
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.autoscale()
    
    info = f"✓ Model architecture uses this\n"
    info += f"✓ graph_net predicts boxes\n"
    info += f"✓ Loss: box_mse (4 coords)\n"
    info += f"✗ Cannot predict polygons"
    ax2.text(0.02, 0.02, info,
            transform=ax2.transAxes, fontsize=10, color='darkblue',
            fontweight='bold', verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    # ===== PANEL 3: Post-Processing Result =====
    ax3 = axes[2]
    ax3.set_title('POST-PROCESSING (MATLAB)\n'
                  'Complex shapes from rectangle operations\n'
                  '(intersect, subtract)',
                  fontsize=11, fontweight='bold', color='purple')
    
    # Boundary
    boundary_polygon = patches.Polygon(
        boundary_coords, fill=True, facecolor='white',
        edgecolor='black', linewidth=3, alpha=0.3
    )
    ax3.add_patch(boundary_polygon)
    
    # Simulate post-processed shapes (approximate)
    # In reality, these come from get_room_boundary.m
    # For visualization, we overlay rectangles with transparency
    for i, box in enumerate(boxes):
        room_type = int(box[0])
        y0, x0, y1, x1 = box[1:5]
        
        color = ROOM_COLORS.get(room_type, '#CCCCCC')
        
        x_min, x_max = min(x0, x1), max(x0, x1)
        y_min, y_max = min(y0, y1), max(y0, y1)
        width = x_max - x_min
        height = y_max - y_min
        
        # Show as rectangles with dashed overlay
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor='purple', linestyle='--',
            facecolor=color, alpha=0.5
        )
        ax3.add_patch(rect)
    
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.autoscale()
    
    info = f"✓ Boolean operations on boxes\n"
    info += f"✓ Creates complex rBoundary\n"
    info += f"✗ NOT same as training data\n"
    info += f"✗ Still based on rectangles"
    ax3.text(0.02, 0.02, info,
            transform=ax3.transAxes, fontsize=10, color='purple',
            fontweight='bold', verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='plum', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    print("\n" + "="*80)
    print("DATA FLOW VISUALIZATION")
    print("Demonstrating where rectangles dominate the pipeline")
    print("="*80)
    
    pkl_path = Path('../Interface/static/Data/data_train_converted.pkl')
    if not pkl_path.exists():
        pkl_path = Path('../../Interface/static/Data/data_train_converted.pkl')
    
    if not pkl_path.exists():
        print("ERROR: Cannot find data_train_converted.pkl")
        return
    
    print(f"\nLoading: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    data = data_dict['data']
    
    # Find good examples
    output_dir = Path('./data_flow_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating data flow visualizations...\n")
    
    count = 0
    for i in range(len(data)):
        sample = data[i]
        rBoundary = sample.rBoundary
        
        # Only pick samples with complex shapes
        has_complex = any(
            rb is not None and len(rb) > 4 
            for rb in rBoundary
        )
        
        if has_complex:
            name = sample.name if hasattr(sample, 'name') else f'sample_{i}'
            output_path = output_dir / f'{count:02d}_{name}_dataflow.png'
            
            print(f"  [{count+1}/5] {name}... ", end='')
            try:
                visualize_data_flow(sample, i, name, output_path)
                print("✓")
                count += 1
                if count >= 5:
                    break
            except Exception as e:
                print(f"✗ {e}")
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")
    print(f"\nOutput: {output_dir.resolve()}")
    print("\nEach image shows 3 panels:")
    print("  LEFT:   Training data with complex polygons (rBoundary)")
    print("  MIDDLE: What model uses - only rectangular boxes")
    print("  RIGHT:  Post-processed output - rectangles with boolean ops")
    print(f"\n{'='*80}")
    print("KEY INSIGHT:")
    print(f"{'='*80}")
    print("The model NEVER learns from the complex polygon shapes.")
    print("It only trains on and generates rectangular bounding boxes.")
    print("This is a fundamental architecture choice, not a bug.")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
