"""
Analyze the rBoundary field to see if rooms are stored as polygons or rectangles.

This will help us understand if the original complex room shapes are preserved
in the processed data or converted to rectangles.
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Room type colors (5-class vocabulary)
ROOM_COLORS = {
    0: '#EE4D4D',  # LivingRoom - Red
    1: '#C67171',  # MasterRoom - Dark Pink
    2: '#FFD274',  # Kitchen - Yellow
    3: '#BEBEBE',  # Bathroom - Gray
    4: '#D11A2D',  # FrontDoor - Dark Red
}

ROOM_NAMES = {
    0: 'LivingRoom',
    1: 'MasterRoom',
    2: 'Kitchen',
    3: 'Bathroom',
    4: 'FrontDoor',
}

def visualize_room_shapes(sample, sample_idx, output_path=None):
    """
    Visualize room shapes to determine if they're polygons or rectangles
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Sample {sample_idx}: Room Shape Analysis', fontsize=16, fontweight='bold')
    
    # Get data
    boundary = sample.boundary if hasattr(sample, 'boundary') else None
    rBoundary = sample.rBoundary if hasattr(sample, 'rBoundary') else None
    boxes = sample.box if hasattr(sample, 'box') else None
    
    print(f"\n{'='*80}")
    print(f"Sample {sample_idx} Analysis:")
    print(f"{'='*80}")
    
    if boundary is not None:
        print(f"Boundary: shape={boundary.shape}")
        boundary_coords = boundary[:, :2] if boundary.shape[1] >= 2 else boundary
        
        # Normalize boundary to reasonable scale for visualization
        if boundary_coords.max() > 100:
            boundary_coords = boundary_coords / boundary_coords.max()
    
    if boxes is not None:
        print(f"Boxes: shape={boxes.shape}")
        # Box format: [room_type, y0, x0, y1, x1] or similar
        print(f"  First box: {boxes[0]}")
    
    if rBoundary is not None:
        print(f"rBoundary: type={type(rBoundary)}, shape={rBoundary.shape if hasattr(rBoundary, 'shape') else 'N/A'}")
        if isinstance(rBoundary, np.ndarray):
            print(f"  Number of room boundaries: {len(rBoundary)}")
            for i, rb in enumerate(rBoundary):
                if rb is not None and hasattr(rb, 'shape'):
                    print(f"  Room {i}: {rb.shape} vertices")
                elif rb is not None:
                    print(f"  Room {i}: type={type(rb)}, len={len(rb) if hasattr(rb, '__len__') else 'N/A'}")
    
    # ===== PLOT 1: Boundary + Bounding Boxes =====
    ax1 = axes[0]
    ax1.set_title('Bounding Boxes (Rectangular)', fontsize=12, fontweight='bold')
    
    # Plot outer boundary
    if boundary is not None and len(boundary) > 0:
        boundary_polygon = patches.Polygon(
            boundary_coords,
            fill=True,
            facecolor='white',
            edgecolor='black',
            linewidth=2
        )
        ax1.add_patch(boundary_polygon)
    
    # Plot bounding boxes
    if boxes is not None and len(boxes) > 0:
        for i, box in enumerate(boxes):
            # Box format: [room_type, y0, x0, y1, x1]
            if len(box) >= 5:
                room_type, y0, x0, y1, x1 = box[:5]
            else:
                continue
            
            room_type_int = int(room_type)
            color = ROOM_COLORS.get(room_type_int, '#CCCCCC')
            room_name = ROOM_NAMES.get(room_type_int, f'Type{room_type_int}')
            
            width = x1 - x0
            height = y1 - y0
            
            rect = patches.Rectangle(
                (x0, y0), width, height,
                linewidth=2,
                edgecolor='black',
                facecolor=color,
                alpha=0.6
            )
            ax1.add_patch(rect)
            ax1.text(x0 + width/2, y0 + height/2, f'{i}\n{room_name}', 
                    ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    
    # ===== PLOT 2: rBoundary (Actual Room Polygons) =====
    ax2 = axes[1]
    ax2.set_title('rBoundary (Actual Shapes)', fontsize=12, fontweight='bold')
    
    # Plot outer boundary
    if boundary is not None and len(boundary) > 0:
        boundary_polygon = patches.Polygon(
            boundary_coords,
            fill=True,
            facecolor='white',
            edgecolor='black',
            linewidth=2
        )
        ax2.add_patch(boundary_polygon)
    
    # Plot room boundaries (actual polygons)
    if rBoundary is not None and boxes is not None:
        for i, (rb, box) in enumerate(zip(rBoundary, boxes)):
            if rb is None:
                continue
            
            # Get room type from box
            room_type_int = int(box[0]) if len(box) > 0 else 0
            color = ROOM_COLORS.get(room_type_int, '#CCCCCC')
            room_name = ROOM_NAMES.get(room_type_int, f'Type{room_type_int}')
            
            # Handle different rBoundary formats
            try:
                if isinstance(rb, np.ndarray):
                    if rb.ndim == 2 and rb.shape[1] >= 2:
                        # Polygon vertices
                        room_coords = rb[:, :2]
                        # Normalize coordinates to prevent huge figure sizes
                        if room_coords.max() > 100:
                            room_coords = room_coords / room_coords.max()
                    elif rb.ndim == 1:
                        # Might be a 1D array that needs reshaping
                        room_coords = rb.reshape(-1, 2)
                        if room_coords.max() > 100:
                            room_coords = room_coords / room_coords.max()
                    else:
                        print(f"  Skipping room {i}: unexpected rBoundary shape {rb.shape}")
                        continue
                else:
                    print(f"  Skipping room {i}: rBoundary is {type(rb)}")
                    continue
                
                # Check if polygon is valid
                if len(room_coords) >= 3:
                    room_polygon = patches.Polygon(
                        room_coords,
                        linewidth=2,
                        edgecolor='black',
                        facecolor=color,
                        alpha=0.6
                    )
                    ax2.add_patch(room_polygon)
                    
                    # Add label at centroid
                    centroid = room_coords.mean(axis=0)
                    ax2.text(centroid[0], centroid[1], f'{i}\n{room_name}', 
                            ha='center', va='center', fontsize=8, fontweight='bold')
                    
                    # Check if polygon is rectangular
                    is_rect = len(room_coords) == 4
                    print(f"  Room {i} ({room_name}): {len(room_coords)} vertices - {'RECTANGLE' if is_rect else 'POLYGON'}")
                    
            except Exception as e:
                print(f"  Error processing room {i}: {e}")
                continue
    
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    # ===== PLOT 3: Overlay Comparison =====
    ax3 = axes[2]
    ax3.set_title('Overlay: Boxes (red) vs Polygons (blue)', fontsize=12, fontweight='bold')
    
    # Plot outer boundary
    if boundary is not None and len(boundary) > 0:
        boundary_polygon = patches.Polygon(
            boundary_coords,
            fill=True,
            facecolor='white',
            edgecolor='black',
            linewidth=2
        )
        ax3.add_patch(boundary_polygon)
    
    # Plot bounding boxes in red
    if boxes is not None and len(boxes) > 0:
        for i, box in enumerate(boxes):
            if len(box) >= 5:
                room_type, y0, x0, y1, x1 = box[:5]
                width = x1 - x0
                height = y1 - y0
                
                rect = patches.Rectangle(
                    (x0, y0), width, height,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none',
                    linestyle='--',
                    alpha=0.8
                )
                ax3.add_patch(rect)
    
    # Plot room polygons in blue
    if rBoundary is not None and boxes is not None:
        for i, (rb, box) in enumerate(zip(rBoundary, boxes)):
            if rb is None:
                continue
            
            try:
                if isinstance(rb, np.ndarray):
                    if rb.ndim == 2 and rb.shape[1] >= 2:
                        room_coords = rb[:, :2]
                    elif rb.ndim == 1:
                        room_coords = rb.reshape(-1, 2)
                    else:
                        continue
                else:
                    continue
                
                if len(room_coords) >= 3:
                    room_polygon = patches.Polygon(
                        room_coords,
                        linewidth=2,
                        edgecolor='blue',
                        facecolor='none',
                        alpha=0.8
                    )
                    ax3.add_patch(room_polygon)
                    
            except Exception as e:
                continue
    
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()
    
    try:
        plt.tight_layout()
    except:
        pass  # Ignore tight_layout warnings
    
    if output_path:
        try:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')  # Lower DPI to avoid size issues
            print(f"\nSaved: {output_path}")
        except Exception as e:
            print(f"\nError saving {output_path}: {e}")
    
    plt.close(fig)


def main():
    print("\n" + "="*80)
    print("ROOM SHAPE ANALYSIS: Polygons vs Rectangles")
    print("="*80 + "\n")
    
    # Load training data
    train_file = Path('../Network/data/data_train.mat')
    output_dir = Path('./room_shape_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not train_file.exists():
        print(f"ERROR: {train_file} not found")
        return
    
    print(f"Loading: {train_file}")
    data = sio.loadmat(train_file, squeeze_me=True, struct_as_record=False)
    samples = data['data']
    
    print(f"Total samples: {len(samples)}")
    
    # Analyze first 10 samples
    num_samples = min(10, len(samples))
    print(f"\nAnalyzing {num_samples} samples...\n")
    
    for i in range(num_samples):
        output_path = output_dir / f'sample_{i:04d}_shape_analysis.png'
        visualize_room_shapes(samples[i], i, output_path)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir.resolve()}")
    print("\nKEY QUESTION ANSWERED:")
    print("  - If rBoundary shows 4-vertex polygons → Rooms are rectangularized")
    print("  - If rBoundary shows >4 vertices → Complex shapes are preserved")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
