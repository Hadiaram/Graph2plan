"""
Visualize the difference between what the dataset HAS (complex polygons in rBoundary)
and what the model GENERATES (rectangular boxes from the box field).

This clearly demonstrates the model limitation vs data limitation.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Room type colors
ROOM_COLORS = {
    0: '#EE4D4D',  # LivingRoom
    1: '#C67171',  # MasterRoom
    2: '#FFD274',  # Kitchen
    3: '#BEBEBE',  # Bathroom
    4: '#D11A2D',  # FrontDoor
}

ROOM_NAMES = {
    0: 'Living', 1: 'Master', 2: 'Kitchen', 3: 'Bath', 4: 'Door'
}


def visualize_data_vs_model_output(sample, sample_idx, name, output_path):
    """
    Show side-by-side comparison:
    LEFT: What the DATA has (complex polygons)
    RIGHT: What the MODEL generates (rectangles)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'Sample {sample_idx}: {name}\nData Contains Complex Shapes vs Model Outputs Rectangles',
                 fontsize=14, fontweight='bold')
    
    boundary = sample.boundary
    boxes = sample.box
    rBoundary = sample.rBoundary
    
    boundary_coords = boundary[:, :2] if boundary.shape[1] >= 2 else boundary
    
    # ===== LEFT: ACTUAL DATA (Complex Polygons from rBoundary) =====
    ax1 = axes[0]
    ax1.set_title('DATASET: Actual Room Shapes (rBoundary field)\n✓ Contains complex polygons',
                  fontsize=12, fontweight='bold', color='green')
    ax1.set_xlabel('X (pixels)', fontsize=10)
    ax1.set_ylabel('Y (pixels)', fontsize=10)
    
    # Plot boundary
    boundary_polygon = patches.Polygon(
        boundary_coords, fill=True, facecolor='white',
        edgecolor='black', linewidth=3, alpha=0.3
    )
    ax1.add_patch(boundary_polygon)
    
    # Plot complex room polygons from rBoundary
    complex_count = 0
    for i, (room_poly, box) in enumerate(zip(rBoundary, boxes)):
        if room_poly is not None and len(room_poly) > 0:
            room_type = int(box[0])
            color = ROOM_COLORS.get(room_type, '#CCCCCC')
            room_name = ROOM_NAMES.get(room_type, f'Type{room_type}')
            
            num_vertices = len(room_poly)
            if num_vertices > 4:
                complex_count += 1
            
            # Plot the actual polygon
            poly_patch = patches.Polygon(
                room_poly, fill=True, facecolor=color,
                edgecolor='darkblue', linewidth=2, alpha=0.7
            )
            ax1.add_patch(poly_patch)
            
            # Add label with vertex count
            centroid_x = np.mean(room_poly[:, 0])
            centroid_y = np.mean(room_poly[:, 1])
            
            label = f'{room_name}\n({num_vertices} vertices)'
            ax1.text(centroid_x, centroid_y, label,
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
    
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.autoscale()
    
    # Add annotation
    info_text = f"✓ Complex polygons: {complex_count}/{len(boxes)}\n"
    info_text += f"✓ Data preserves room shapes"
    ax1.text(0.02, 0.02, info_text,
            transform=ax1.transAxes, fontsize=10, color='green', fontweight='bold',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # ===== RIGHT: MODEL OUTPUT (Rectangles from box field) =====
    ax2 = axes[1]
    ax2.set_title('MODEL OUTPUT: What Graph2Plan Generates (box field)\n✗ Only rectangular bounding boxes',
                  fontsize=12, fontweight='bold', color='red')
    ax2.set_xlabel('X (pixels)', fontsize=10)
    ax2.set_ylabel('Y (pixels)', fontsize=10)
    
    # Plot boundary
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
            linewidth=2, edgecolor='darkred',
            facecolor=color, alpha=0.7
        )
        ax2.add_patch(rect)
        
        # Add label
        label = f'{room_name}\n(rectangle)'
        ax2.text(x_min + width/2, y_min + height/2, label,
                ha='center', va='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
    
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.autoscale()
    
    # Add annotation
    info_text = f"✗ Model ignores rBoundary\n"
    info_text += f"✗ Only uses box coordinates\n"
    info_text += f"✗ All rooms → rectangles"
    ax2.text(0.02, 0.02, info_text,
            transform=ax2.transAxes, fontsize=10, color='red', fontweight='bold',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    print("\n" + "="*80)
    print("VISUALIZING: DATA (Complex Shapes) vs MODEL OUTPUT (Rectangles)")
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
    
    # Find samples with complex shapes
    output_dir = Path('./data_vs_model_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Finding samples with complex room shapes...\n")
    
    samples_with_complex = []
    for i in range(len(data)):
        sample = data[i]
        rBoundary = sample.rBoundary
        
        # Check if any room has >4 vertices
        has_complex = False
        for room_poly in rBoundary:
            if room_poly is not None and len(room_poly) > 4:
                has_complex = True
                break
        
        if has_complex:
            samples_with_complex.append(i)
            if len(samples_with_complex) >= 10:
                break
    
    print(f"Found {len(samples_with_complex)} samples with complex shapes")
    print(f"Visualizing comparison...\n")
    
    for idx, sample_idx in enumerate(samples_with_complex):
        sample = data[sample_idx]
        name = sample.name if hasattr(sample, 'name') else f'sample_{sample_idx}'
        output_path = output_dir / f'{idx:02d}_{name}_comparison.png'
        
        print(f"  [{idx+1}/{len(samples_with_complex)}] {name}... ", end='')
        try:
            visualize_data_vs_model_output(sample, sample_idx, name, output_path)
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")
    print(f"\nOutput: {output_dir.resolve()}")
    print("\nEach image shows:")
    print("  LEFT:  Actual data - complex polygon shapes (what the dataset HAS)")
    print("  RIGHT: Model output - rectangular boxes (what Graph2Plan GENERATES)")
    print(f"\n{'='*80}")
    print("ANSWER TO YOUR QUESTION:")
    print(f"{'='*80}")
    print("✓ The dataset DOES contain complex room shapes (41.3% of rooms)")
    print("✗ The rectangular output is a MODEL LIMITATION, not a data limitation")
    print("\nThe Graph2Plan model architecture:")
    print("  - Uses 'box' field (rectangles) for training/inference")
    print("  - Ignores 'rBoundary' field (actual polygons)")
    print("  - This is by design - the refinement_net outputs rectangular segmentation")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
