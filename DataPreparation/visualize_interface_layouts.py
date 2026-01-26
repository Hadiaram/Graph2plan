"""
Visualize floor plan layouts from Interface pkl files (data_train_converted.pkl, data_test_converted.pkl)

These are the actual files used by the Graph2Plan model for training and inference.
This script will show:
1. Boundary outline
2. Bounding boxes (if available)
3. Layout segmentation map (the actual generated layout)

Usage:
    python visualize_interface_layouts.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm import tqdm

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

def visualize_sample(sample, sample_idx, name, output_path=None):
    """
    Visualize a single floor plan sample from pkl file
    
    Args:
        sample: A single data sample (dict or object)
        sample_idx: Index of the sample
        name: Name of the floor plan
        output_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Sample {sample_idx}: {name}', fontsize=16, fontweight='bold')
    
    # Helper to get attributes
    def get_attr(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        else:
            return getattr(obj, key, None)
    
    # Get data fields
    boundary = get_attr(sample, 'boundary')
    boxes = get_attr(sample, 'boxes') or get_attr(sample, 'box')
    objs = get_attr(sample, 'objs') or get_attr(sample, 'room_types')
    layout = get_attr(sample, 'layout')
    
    # ===== PLOT 1: Boundary Only =====
    ax1 = axes[0]
    ax1.set_title('Boundary', fontsize=12, fontweight='bold')
    
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
    
    # Plot boxes
    if boxes is not None and objs is not None and len(boxes) > 0:
        for i, (box, room_type) in enumerate(zip(boxes, objs)):
            room_type_int = int(room_type)
            color = ROOM_COLORS.get(room_type_int, '#CCCCCC')
            room_name = ROOM_NAMES.get(room_type_int, f'Type{room_type_int}')
            
            # Box format: [y0, x0, y1, x1] or similar
            if len(box) >= 4:
                y0, x0, y1, x1 = box[:4]
            else:
                continue
            
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
    
    # ===== PLOT 3: Layout Segmentation =====
    ax3 = axes[2]
    ax3.set_title('Layout Segmentation', fontsize=12, fontweight='bold')
    
    if layout is not None and len(layout) > 0:
        # Check if layout is valid
        if isinstance(layout, np.ndarray):
            # Create RGB image from layout indices
            h, w = layout.shape[:2] if layout.ndim >= 2 else (128, 128)
            
            if layout.ndim == 2:
                # Layout is 2D segmentation map
                layout_rgb = np.ones((h, w, 3))
                
                # Color each room type
                for room_idx in range(6):  # 0-4 rooms + 5 background
                    mask = (layout == room_idx)
                    if mask.any():
                        hex_color = ROOM_COLORS.get(room_idx, '#CCCCCC')
                        rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16)/255.0 for i in (0, 2, 4))
                        layout_rgb[mask] = rgb
                
                ax3.imshow(layout_rgb, origin='upper', interpolation='nearest')
                ax3.axis('off')
                
                # Add info
                unique_vals = np.unique(layout)
                room_counts = {int(v): np.sum(layout == v) for v in unique_vals}
                info_text = f"Shape: {layout.shape}\n"
                info_text += f"Unique: {sorted([int(v) for v in unique_vals])}\n"
                info_text += f"Rooms: {len([v for v in unique_vals if v < 5])}"
                
                ax3.text(0.02, 0.98, info_text,
                        transform=ax3.transAxes,
                        fontsize=9,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            else:
                ax3.text(0.5, 0.5, f'Layout has unexpected shape: {layout.shape}',
                        ha='center', va='center', transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, f'Layout type: {type(layout)}',
                    ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, 'No layout data',
                ha='center', va='center', transform=ax3.transAxes,
                fontsize=14, color='red')
        ax3.axis('off')
    
    try:
        plt.tight_layout()
    except:
        pass
    
    if output_path:
        try:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            print(f"  Saved: {output_path.name}")
        except Exception as e:
            print(f"  Error saving {output_path.name}: {e}")
    else:
        plt.show()
    
    plt.close(fig)


def analyze_pkl_file(pkl_path, output_dir, num_samples=10):
    """
    Load a pkl file and visualize samples
    
    Args:
        pkl_path: Path to pkl file
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    print("\n" + "="*80)
    print(f"Analyzing: {pkl_path}")
    print("="*80)
    
    if not pkl_path.exists():
        print(f"ERROR: File not found: {pkl_path}")
        return
    
    # Load pkl file
    print(f"Loading pickle file...")
    with open(pkl_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    # Inspect structure
    print(f"\nPickle file keys: {list(data_dict.keys())}")
    
    # Get data
    data = data_dict.get('data')
    if data is None:
        print("ERROR: No 'data' key in pickle file")
        print(f"Available keys: {list(data_dict.keys())}")
        return
    
    # Get name list
    name_list = data_dict.get('testNameList', data_dict.get('trainNameList', []))
    if isinstance(name_list, np.ndarray) and len(name_list) > 0:
        name_list = [str(n).strip() for n in name_list]
    elif not isinstance(name_list, list):
        name_list = []
    
    print(f"Data type: {type(data)}")
    print(f"Number of samples: {len(data)}")
    print(f"Number of names: {len(name_list)}")
    
    # Check first sample structure
    if len(data) > 0:
        first = data[0]
        print(f"\nFirst sample type: {type(first)}")
        
        if isinstance(first, dict):
            print(f"First sample keys: {list(first.keys())}")
            for key, val in first.items():
                if hasattr(val, 'shape'):
                    print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
                else:
                    print(f"  {key}: type={type(val)}")
        else:
            attrs = [a for a in dir(first) if not a.startswith('_')]
            print(f"First sample attributes: {attrs[:10]}")
            for attr in attrs[:10]:
                try:
                    val = getattr(first, attr)
                    if hasattr(val, 'shape'):
                        print(f"  {attr}: shape={val.shape}, dtype={val.dtype}")
                    else:
                        print(f"  {attr}: type={type(val)}")
                except:
                    pass
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize samples
    print(f"\nVisualizing {num_samples} samples...")
    for i in range(min(num_samples, len(data))):
        name = name_list[i] if i < len(name_list) else f"sample_{i}"
        output_path = output_dir / f'{i:04d}_{name}.png'
        
        try:
            visualize_sample(data[i], i, name, output_path)
        except Exception as e:
            print(f"  Error processing sample {i} ({name}): {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"COMPLETE: Visualized {min(num_samples, len(data))} samples")
    print(f"Output: {output_dir.resolve()}")
    print(f"{'='*80}")


def main():
    print("\n" + "="*80)
    print("INTERFACE LAYOUT VISUALIZATION")
    print("Visualizing pkl files from Interface/static/Data")
    print("="*80)
    
    # Paths - check multiple possible locations
    interface_base = Path('../Interface/static/Data')
    
    # Alternative path if running from different location
    if not interface_base.exists():
        interface_base = Path('../../Interface/static/Data')
    
    if not interface_base.exists():
        print(f"ERROR: Interface data directory not found")
        print(f"Tried: ../Interface/static/Data")
        return
    
    output_base = Path('./layout_visualizations')
    
    # Find pkl files
    train_pkl = interface_base / 'data_train_converted.pkl'
    test_pkl = interface_base / 'data_test_converted.pkl'
    
    if not train_pkl.exists() and not test_pkl.exists():
        print(f"ERROR: No pkl files found in {interface_base}")
        print(f"Looking for:")
        print(f"  - data_train_converted.pkl")
        print(f"  - data_test_converted.pkl")
        return
    
    # Analyze training data
    if train_pkl.exists():
        print("\n>>> TRAINING DATA <<<")
        analyze_pkl_file(train_pkl, output_base / 'train', num_samples=10)
    
    # Analyze test data
    if test_pkl.exists():
        print("\n>>> TEST DATA <<<")
        analyze_pkl_file(test_pkl, output_base / 'test', num_samples=10)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nCheck the output directories for visualizations:")
    print(f"  {(output_base / 'train').resolve()}")
    print(f"  {(output_base / 'test').resolve()}")
    print("\nEach image shows:")
    print("  - Left: Boundary outline")
    print("  - Middle: Bounding boxes with room labels")
    print("  - Right: Layout segmentation map (if available)")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
