"""
Generate PNG images for ResPlan floor plans

This script creates visual thumbnails of floor plans from the data_train_converted.pkl
and data_test_converted.pkl files for display in the Interface.

Usage:
    python generate_floorplan_images.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm.auto import tqdm

# Room type colors (matching Graph2plan's color scheme)
ROOM_COLORS = {
    0: '#EE4D4D',  # LivingRoom - Red
    1: '#C67171',  # MasterRoom - Dark Pink
    2: '#FFD274',  # Kitchen - Yellow
    3: '#BEBEBE',  # Bathroom - Gray
    4: '#BFE3E8',  # DiningRoom - Light Blue
    5: '#7BA779',  # ChildRoom - Green
    6: '#E87A90',  # StudyRoom - Pink
    7: '#FF8C69',  # SecondRoom - Orange
    8: '#1F849B',  # GuestRoom - Dark Blue
    9: '#727171',  # Balcony - Dark Gray
    10: '#785A67', # Entrance - Purple
    11: '#D3A2C7', # Storage - Light Purple
    12: '#FFFFFF', # Wall-in - White
    13: '#FFFF00', # External - Yellow
    14: '#FFFFFF', # ExteriorWall - White
    15: '#D11A2D', # FrontDoor - Dark Red
    16: '#1F849B', # InteriorWall - Dark Blue
    17: '#BEBEBE', # InteriorDoor - Gray
}

def plot_floorplan(boundary, boxes, room_types, output_path, dpi=100):
    """
    Render a floor plan as a PNG image

    Args:
        boundary: Nx2 array of boundary coordinates
        boxes: Nx4 array of room bounding boxes (y0, x0, y1, x1)
        room_types: N array of room type indices
        output_path: Path to save PNG file
        dpi: Image resolution (default 100)
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=dpi)

    # Get boundary coordinates
    if isinstance(boundary, np.ndarray):
        if boundary.shape[1] >= 2:
            boundary_coords = boundary[:, :2]
        else:
            boundary_coords = boundary
    else:
        boundary_coords = np.array(boundary)

    # Plot boundary (filled with white background)
    boundary_polygon = patches.Polygon(
        boundary_coords,
        fill=True,
        facecolor='white',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(boundary_polygon)

    # Room rendering disabled - only showing boundary outline
    # (Uncomment below if you want to render individual rooms)
    # for i, (box, rtype) in enumerate(zip(boxes, room_types)):
    #     rtype_int = int(rtype)
    #     if len(box) >= 4:
    #         y0, x0, y1, x1 = box[:4]
    #     else:
    #         continue
    #     width = x1 - x0
    #     height = y1 - y0
    #     color = ROOM_COLORS.get(rtype_int, '#CCCCCC')
    #     rect = patches.Rectangle(
    #         (x0, y0), width, height,
    #         linewidth=1,
    #         edgecolor='black',
    #         facecolor=color,
    #         alpha=0.7
    #     )
    #     ax.add_patch(rect)

    # Set axis properties
    ax.set_aspect('equal')
    ax.autoscale()
    ax.invert_yaxis()  # Flip y-axis to match coordinate system
    ax.axis('off')

    # Save figure
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)


def generate_images_from_pkl(pkl_path, output_dir, subset_name):
    """
    Generate PNG images for all floor plans in a PKL file

    Args:
        pkl_path: Path to data_train_converted.pkl or data_test_converted.pkl
        output_dir: Directory to save PNG images
        subset_name: 'train' or 'test' for progress display
    """
    print(f"\nLoading {subset_name} data from: {pkl_path}")

    # Load data
    data_dict = pickle.load(open(pkl_path, 'rb'))
    data = data_dict['data']

    # Extract name list for proper image naming
    if subset_name == 'test':
        name_list = data_dict.get('testNameList', [])
    else:
        name_list = data_dict.get('trainNameList', [])

    # Convert to list and strip spaces
    if isinstance(name_list, np.ndarray):
        name_list = [str(n).strip() for n in name_list]
    else:
        name_list = [str(n).strip() for n in name_list]

    # Debug: print data structure
    print(f"  Data type: {type(data)}")
    if isinstance(data, np.ndarray):
        print(f"  Array shape: {data.shape}, dtype: {data.dtype}")
    print(f"  Name list: {len(name_list)} names (first 5: {name_list[:5]})")

    # Handle scipy.io.loadmat squeeze_me=True squeezing arrays
    if not isinstance(data, (list, np.ndarray)):
        # Single item squeezed to scalar - convert to list
        data = [data]
    elif isinstance(data, np.ndarray):
        # NumPy array - convert to list
        if data.ndim == 0:
            # 0-dimensional array (squeezed) - extract item and wrap
            data = [data.item()]
        elif data.ndim == 1:
            # 1D array - this is the normal case for struct arrays
            data = list(data)
        else:
            # Multi-dimensional array - flatten to list
            data = list(data.flat)

    print(f"  Found {len(data)} floor plans")
    print(f"Generating PNG images to: {output_dir}")

    # Create output directory
    output_dir = Path(output_dir)  # Convert string to Path object
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate images
    for i, item in enumerate(tqdm(data, desc=f"Generating {subset_name} images")):
        try:
            # Handle both dict and object access
            def get_attr(obj, key):
                if isinstance(obj, dict):
                    return obj.get(key)
                else:
                    return getattr(obj, key, None)

            # Get floor plan name from name_list
            if i < len(name_list):
                name = name_list[i]
            else:
                name = str(i)  # Fallback to index if name_list is short

            # Get floor plan data
            boundary = get_attr(item, 'boundary')

            # Check boundary exists
            if boundary is None or len(boundary) == 0:
                print(f"Warning: No boundary for {name}, skipping")
                continue

            # For boundary-only rendering, we don't need box data
            # Use empty arrays since room rendering is disabled
            boxes = np.array([])
            room_types = np.array([])

            # Generate image
            output_path = output_dir / f"{name}.png"
            plot_floorplan(boundary, boxes, room_types, output_path)

        except Exception as e:
            print(f"Error processing floor plan {i} (name={name if 'name' in locals() else 'unknown'}): {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"SUCCESS: Generated {len(list(output_dir.glob('*.png')))} images for {subset_name} set")


def main():
    # Paths
    interface_img_dir = Path('../Interface/static/Data/Img')
    train_pkl = Path(r'C:\Users\hmbashir\source\Graph2plan\Interface\static\Data\data_train_converted.pkl')
    test_pkl = Path('./data/data_test_converted.pkl')

    print("=" * 60)
    print("ResPlan Floor Plan Image Generator - TEST SET ONLY")
    print("=" * 60)

    # Check if test file exists
    if not test_pkl.exists():
        # Try Interface location
        test_pkl = Path('../Interface/static/Data/data_test_converted.pkl')

        if not test_pkl.exists():
            print(f"Error: Could not find data_test_converted.pkl")
            print(f"Tried:")
            print(f"  - ./data/data_test_converted.pkl")
            print(f"  - ../Interface/static/Data/data_test_converted.pkl")
            return

    # SKIP training set generation - Interface uses test data only
    print("INFO: Skipping train set (Interface uses test data)")

    # Generate images for test set ONLY
    if test_pkl.exists():
        generate_images_from_pkl(test_pkl, interface_img_dir, 'test')
    else:
        print(f"Error: {test_pkl} not found")

    if train_pkl.exists():
        generate_images_from_pkl(train_pkl, r"C:\Users\hmbashir\Documents\Prompt_to_Graph\Debugging\Image Comparison\Processed", 'train')
    else:
        print(f"Error: {train_pkl} not found")


    print("\n" + "=" * 60)
    print("SUCCESS: Image generation complete!")
    print(f"Images saved to: {interface_img_dir.resolve()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
