"""
Visualize processed floor plan data to check if room shapes are rectangular or complex.

This script loads the processed .mat files (data_train.mat, data_test.mat) and 
visualizes the actual room data to determine:
1. Are rooms stored as rectangular bounding boxes only?
2. Or do they preserve complex/irregular shapes from the original data?

Usage:
    python visualize_processed_data.py
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

def visualize_floorplan_from_mat(sample, sample_idx, output_path=None):
    """
    Visualize a single floor plan sample from the .mat file
    
    Args:
        sample: A single data sample (struct from .mat file)
        sample_idx: Index of the sample (for title)
        output_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # ===== PLOT 1: Boundary only =====
    ax1 = axes[0]
    ax1.set_title(f'Sample {sample_idx}: Boundary Only', fontsize=14, fontweight='bold')
    
    boundary = sample.boundary if hasattr(sample, 'boundary') else None
    if boundary is not None and len(boundary) > 0:
        # Plot boundary
        boundary_coords = boundary[:, :2] if boundary.shape[1] >= 2 else boundary
        boundary_polygon = patches.Polygon(
            boundary_coords,
            fill=False,
            edgecolor='black',
            linewidth=2
        )
        ax1.add_patch(boundary_polygon)
        ax1.plot(boundary_coords[:, 0], boundary_coords[:, 1], 'ko-', markersize=3, linewidth=2, label='Boundary')
    
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # ===== PLOT 2: Room Bounding Boxes =====
    ax2 = axes[1]
    ax2.set_title(f'Sample {sample_idx}: Room Bounding Boxes', fontsize=14, fontweight='bold')
    
    # Plot boundary
    if boundary is not None and len(boundary) > 0:
        boundary_polygon = patches.Polygon(
            boundary_coords,
            fill=True,
            facecolor='white',
            edgecolor='black',
            linewidth=2,
            alpha=0.3
        )
        ax2.add_patch(boundary_polygon)
    
    # Get room data - handle different field name conventions
    boxes_field = sample.box if hasattr(sample, 'box') else (sample.boxes if hasattr(sample, 'boxes') else None)
    
    if boxes_field is not None and len(boxes_field) > 0:
        print(f"\nSample {sample_idx} room data:")
        print(f"  Number of rooms: {len(boxes_field)}")
        print(f"  Boxes shape: {boxes_field.shape}")
        print(f"  Boxes dtype: {boxes_field.dtype}")
        
        # Box format: [room_type, y0, x0, y1, x1] in pixel coordinates
        for i, box_row in enumerate(boxes_field):
            if len(box_row) >= 5:
                room_type_int = int(box_row[0])
                y0, x0, y1, x1 = box_row[1:5]
            else:
                continue
                
            color = ROOM_COLORS.get(room_type_int, '#CCCCCC')
            room_name = ROOM_NAMES.get(room_type_int, f'Type{room_type_int}')
            
            width = abs(x1 - x0)
            height = abs(y1 - y0)
            
            print(f"  Room {i}: {room_name} - Box: [{y0:.1f}, {x0:.1f}, {y1:.1f}, {x1:.1f}] (w={width:.1f}, h={height:.1f})")
            
            # Draw rectangle - coordinates need to use min/max to handle any order
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
                    ha='center', va='center', fontsize=8, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No box data', ha='center', va='center', transform=ax2.transAxes)
    
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.autoscale()  # Auto-scale instead of fixed limits
    ax2.invert_yaxis()
    
    # ===== PLOT 3: Layout (Segmentation Map) =====
    ax3 = axes[2]
    ax3.set_title(f'Sample {sample_idx}: Layout Segmentation', fontsize=14, fontweight='bold')
    
    layout = sample.layout if hasattr(sample, 'layout') else None
    if layout is not None and len(layout) > 0:
        print(f"  Layout shape: {layout.shape}")
        print(f"  Layout unique values: {np.unique(layout)}")
        
        # Create color map for segmentation
        # Layout values: 0-4 are room types, 5 is background/boundary
        segmentation_colors = np.zeros((*layout.shape, 3))
        
        for room_idx in range(5):
            mask = (layout == room_idx)
            if mask.any():
                hex_color = ROOM_COLORS.get(room_idx, '#CCCCCC')
                rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16)/255.0 for i in (0, 2, 4))
                segmentation_colors[mask] = rgb
        
        # Background (index 5) = white
        background_mask = (layout == 5)
        segmentation_colors[background_mask] = [1, 1, 1]
        
        ax3.imshow(segmentation_colors, origin='upper')
        ax3.axis('off')
    else:
        ax3.text(0.5, 0.5, 'No layout data', ha='center', va='center', transform=ax3.transAxes)
    
    try:
        plt.tight_layout()
    except:
        pass  # Ignore layout warnings
    
    if output_path:
        try:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')  # Reduced DPI
            print(f"\nSaved visualization to: {output_path}")
        except Exception as e:
            print(f"\nError saving {output_path}: {e}")
            # Try without bbox_inches
            try:
                plt.savefig(output_path, dpi=100)
                print(f"Saved with default bounds: {output_path}")
            except:
                print(f"Could not save {output_path}")
    else:
        plt.show()
    
    plt.close(fig)


def analyze_data_file(mat_path, output_dir, num_samples=5):
    """
    Load a .mat file and visualize several samples
    
    Args:
        mat_path: Path to data_train.mat or data_test.mat
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    print("="*80)
    print(f"Analyzing: {mat_path}")
    print("="*80)
    
    # Load data
    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    samples = data['data']
    
    print(f"\nDataset info:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Sample type: {type(samples[0])}")
    
    # Check first sample structure
    first = samples[0]
    print(f"\nFirst sample attributes:")
    attrs = [attr for attr in dir(first) if not attr.startswith('_')]
    for attr in attrs:
        val = getattr(first, attr)
        if hasattr(val, 'shape'):
            print(f"  {attr}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  {attr}: type={type(val)}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize samples
    print(f"\nVisualizing {num_samples} samples...")
    for i in range(min(num_samples, len(samples))):
        output_path = output_dir / f'sample_{i:04d}.png'
        visualize_floorplan_from_mat(samples[i], i, output_path)
    
    print(f"\n{'='*80}")
    print(f"COMPLETE: Visualized {min(num_samples, len(samples))} samples")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"{'='*80}")


def main():
    print("\n" + "="*80)
    print("PROCESSED DATA VISUALIZATION TOOL")
    print("Checking if room shapes are rectangular or complex")
    print("="*80 + "\n")
    
    # Paths
    network_data_dir = Path('../Network/data')
    output_base_dir = Path('./visualizations')
    
    # Check which files exist
    train_file = network_data_dir / 'data_train.mat'
    test_file = network_data_dir / 'data_test.mat'
    
    if not train_file.exists() and not test_file.exists():
        print("ERROR: No data files found!")
        print(f"  Checked: {network_data_dir.resolve()}")
        return
    
    # Analyze training data
    if train_file.exists():
        print("\n>>> ANALYZING TRAINING DATA <<<")
        analyze_data_file(
            train_file, 
            output_base_dir / 'train',
            num_samples=10
        )
    
    # Analyze test data
    if test_file.exists():
        print("\n>>> ANALYZING TEST DATA <<<")
        analyze_data_file(
            test_file,
            output_base_dir / 'test',
            num_samples=10
        )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nKEY FINDINGS TO CHECK:")
    print("  1. Are 'boxes' simple rectangles [y0, x0, y1, x1]?")
    print("  2. Does 'layout' (segmentation map) show rectangular or complex shapes?")
    print("  3. Does 'boundary' represent the outer perimeter only?")
    print("\nIf layout shows rectangular shapes but original data had complex shapes,")
    print("then the Graph2Plan preprocessing converts complex rooms to rectangles.")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
