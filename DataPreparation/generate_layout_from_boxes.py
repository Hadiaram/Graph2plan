"""
Generate layout segmentation maps from bounding boxes in Interface pkl files.

This replicates what the Graph2Plan model does - converting bounding boxes
into a pixel-wise segmentation map showing room layouts.

Usage:
    python generate_layout_from_boxes.py
"""

import pickle
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
    5: '#FFFFFF',  # Background/Boundary - White
}

ROOM_NAMES = {
    0: 'Living',
    1: 'Master',
    2: 'Kitchen',
    3: 'Bath',
    4: 'Door',
    5: 'Background',
}


def create_layout_from_boxes(boxes, objs, layout_size=(256, 256)):
    """
    Create a layout segmentation map from bounding boxes.
    
    Args:
        boxes: Array of shape (N, 5) with format [room_type, y0, x0, y1, x1]
        objs: Array of room types (not used if room_type is in boxes)
        layout_size: Output size (height, width)
    
    Returns:
        layout: 2D array of shape layout_size with room type indices
    """
    h, w = layout_size
    layout = np.ones((h, w), dtype=np.uint8) * 5  # Initialize with background (5)
    
    # Process boxes in order (later boxes may overlap earlier ones)
    for i, box in enumerate(boxes):
        # Parse box: [room_type, y0, x0, y1, x1]
        room_type = int(box[0])
        y0, x0, y1, x1 = box[1:5]
        
        # Convert to pixel coordinates
        y0_px = int(np.clip(y0, 0, h-1))
        y1_px = int(np.clip(y1, 0, h-1))
        x0_px = int(np.clip(x0, 0, w-1))
        x1_px = int(np.clip(x1, 0, w-1))
        
        # Ensure proper ordering
        y_min, y_max = min(y0_px, y1_px), max(y0_px, y1_px)
        x_min, x_max = min(x0_px, x1_px), max(x0_px, x1_px)
        
        # Fill the rectangle
        if y_max > y_min and x_max > x_min:
            layout[y_min:y_max, x_min:x_max] = room_type
    
    return layout


def visualize_sample_with_layout(sample, sample_idx, name, output_path):
    """
    Visualize a floor plan with generated layout segmentation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Sample {sample_idx}: {name}', fontsize=16, fontweight='bold')
    
    # Get data
    boundary = sample.boundary if hasattr(sample, 'boundary') else None
    boxes = sample.box if hasattr(sample, 'box') else None
    
    # ===== PLOT 1: Boundary Only =====
    ax1 = axes[0]
    ax1.set_title('Boundary', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (pixels)', fontsize=10)
    ax1.set_ylabel('Y (pixels)', fontsize=10)
    
    if boundary is not None and len(boundary) > 0:
        boundary_coords = boundary[:, :2] if boundary.shape[1] >= 2 else boundary
        boundary_polygon = patches.Polygon(
            boundary_coords,
            fill=True,
            facecolor='white',
            edgecolor='black',
            linewidth=2
        )
        ax1.add_patch(boundary_polygon)
        ax1.plot(boundary_coords[:, 0], boundary_coords[:, 1], 'ko-', markersize=2, linewidth=1)
    
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.autoscale()
    
    # ===== PLOT 2: Bounding Boxes =====
    ax2 = axes[1]
    ax2.set_title('Bounding Boxes', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (pixels)', fontsize=10)
    ax2.set_ylabel('Y (pixels)', fontsize=10)
    
    # Plot boundary
    if boundary is not None and len(boundary) > 0:
        boundary_polygon = patches.Polygon(
            boundary_coords,
            fill=True,
            facecolor='#F5F5F5',
            edgecolor='black',
            linewidth=2,
            alpha=0.3
        )
        ax2.add_patch(boundary_polygon)
    
    num_rooms = 0
    # Plot boxes
    if boxes is not None and len(boxes) > 0:
        num_rooms = len(boxes)
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
                linewidth=2,
                edgecolor='black',
                facecolor=color,
                alpha=0.6
            )
            ax2.add_patch(rect)
            
            # Add label
            ax2.text(x_min + width/2, y_min + height/2, room_name,
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.autoscale()
    
    # ===== PLOT 3: Generated Layout =====
    ax3 = axes[2]
    ax3.set_title('Generated Layout (from boxes)', fontsize=12, fontweight='bold')
    
    if boxes is not None and len(boxes) > 0:
        # Generate layout from boxes
        layout = create_layout_from_boxes(boxes, None, layout_size=(256, 256))
        
        # Create RGB image
        h, w = layout.shape
        layout_rgb = np.ones((h, w, 3))
        
        for room_idx in range(6):  # 0-4 rooms + 5 background
            mask = (layout == room_idx)
            if mask.any():
                hex_color = ROOM_COLORS.get(room_idx, '#CCCCCC')
                rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16)/255.0 for i in (0, 2, 4))
                layout_rgb[mask] = rgb
        
        ax3.imshow(layout_rgb, origin='upper', interpolation='nearest')
        ax3.axis('off')
        
        # Add statistics
        unique_vals = np.unique(layout)
        room_counts = {int(v): np.sum(layout == v) for v in unique_vals}
        
        info_text = f"Size: {layout.shape}\n"
        info_text += f"Rooms: {num_rooms}\n"
        info_text += "Pixels per room:\n"
        for room_idx in sorted(room_counts.keys()):
            if room_idx < 5:  # Skip background
                room_name = ROOM_NAMES.get(room_idx, f'Type{room_idx}')
                pixels = room_counts[room_idx]
                info_text += f"  {room_name}: {pixels}\n"
        
        ax3.text(0.02, 0.98, info_text,
                transform=ax3.transAxes,
                fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    else:
        ax3.text(0.5, 0.5, 'No boxes available',
                ha='center', va='center', transform=ax3.transAxes,
                fontsize=14, color='red')
        ax3.axis('off')
    
    try:
        plt.tight_layout()
    except:
        pass
    
    try:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        return True
    except Exception as e:
        print(f"  Error saving {output_path.name}: {e}")
        return False
    finally:
        plt.close(fig)


def main():
    print("\n" + "="*80)
    print("LAYOUT GENERATION FROM BOUNDING BOXES")
    print("Converting bounding boxes to segmentation maps")
    print("="*80)
    
    # Load training data
    pkl_path = Path('../Interface/static/Data/data_train_converted.pkl')
    
    if not pkl_path.exists():
        pkl_path = Path('../../Interface/static/Data/data_train_converted.pkl')
    
    if not pkl_path.exists():
        print(f"ERROR: Cannot find data_train_converted.pkl")
        return
    
    print(f"\nLoading: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    data = data_dict['data']
    print(f"Total samples: {len(data)}")
    
    # Create output directory
    output_dir = Path('./generated_layouts')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process samples
    num_samples = 20
    print(f"\nGenerating layouts for {num_samples} samples...")
    
    success_count = 0
    for i in range(min(num_samples, len(data))):
        sample = data[i]
        name = sample.name if hasattr(sample, 'name') else f'sample_{i}'
        output_path = output_dir / f'{i:04d}_{name}.png'
        
        print(f"  [{i+1}/{num_samples}] {name}...", end=' ')
        
        try:
            if visualize_sample_with_layout(sample, i, name, output_path):
                print(f"✓ Saved")
                success_count += 1
            else:
                print(f"✗ Failed")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\n{'='*80}")
    print(f"COMPLETE: Generated {success_count}/{num_samples} layouts")
    print(f"Output: {output_dir.resolve()}")
    print(f"{'='*80}")
    print("\nEach image shows:")
    print("  - Left: Boundary outline")
    print("  - Middle: Bounding boxes with room labels")
    print("  - Right: Generated layout segmentation (what Graph2Plan outputs)")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
