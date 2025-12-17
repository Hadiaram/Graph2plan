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

    # Plot boundary
    boundary_polygon = patches.Polygon(
        boundary_coords,
        fill=False,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(boundary_polygon)

    # Plot rooms
    for i, (box, rtype) in enumerate(zip(boxes, room_types)):
        rtype_int = int(rtype)

        # Box format: (y0, x0, y1, x1) -> convert to matplotlib (x0, y0, width, height)
        if len(box) >= 4:
            y0, x0, y1, x1 = box[:4]
        else:
            continue

        width = x1 - x0
        height = y1 - y0

        # Get room color
        color = ROOM_COLORS.get(rtype_int, '#CCCCCC')

        # Create rectangle
        rect = patches.Rectangle(
            (x0, y0), width, height,
            linewidth=1,
            edgecolor='black',
            facecolor=color,
            alpha=0.7
        )
        ax.add_patch(rect)

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

    if not isinstance(data, list):
        data = [data]

    print(f"Found {len(data)} floor plans")
    print(f"Generating PNG images to: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate images
    for i, item in enumerate(tqdm(data, desc=f"Generating {subset_name} images")):
        try:
            # Get floor plan name
            if hasattr(item, 'name'):
                name = str(item.name)
            else:
                name = str(i)

            # Get floor plan data
            boundary = item.boundary if hasattr(item, 'boundary') else None

            # Get boxes - may be in 'box' attribute or need to construct from gtBox
            if hasattr(item, 'box'):
                boxes = item.box[:, :4] if item.box.shape[1] > 4 else item.box
                room_types = item.box[:, -1] if item.box.shape[1] > 4 else np.zeros(len(item.box))
            elif hasattr(item, 'gtBox'):
                boxes = item.gtBox
                room_types = item.rType if hasattr(item, 'rType') else np.zeros(len(boxes))
            else:
                print(f"Warning: No box data for {name}, skipping")
                continue

            if boundary is None or len(boundary) == 0:
                print(f"Warning: No boundary for {name}, skipping")
                continue

            # Generate image
            output_path = output_dir / f"{name}.png"
            plot_floorplan(boundary, boxes, room_types, output_path)

        except Exception as e:
            print(f"Error processing floor plan {i} ({name}): {e}")
            continue

    print(f"✓ Generated {len(list(output_dir.glob('*.png')))} images for {subset_name} set")


def main():
    # Paths
    interface_img_dir = Path('../Interface/static/Data/Img')
    train_pkl = Path('./data/data_train_converted.pkl')
    test_pkl = Path('./data/data_test_converted.pkl')

    print("=" * 60)
    print("ResPlan Floor Plan Image Generator")
    print("=" * 60)

    # Check if files exist
    if not train_pkl.exists():
        # Try Interface location
        train_pkl = Path('../Interface/static/Data/data_train_converted.pkl')
        test_pkl = Path('../Interface/static/Data/data_test_converted.pkl')

        if not train_pkl.exists():
            print(f"Error: Could not find data_train_converted.pkl")
            print(f"Tried:")
            print(f"  - ./data/data_train_converted.pkl")
            print(f"  - ../Interface/static/Data/data_train_converted.pkl")
            return

    # Generate images for training set
    if train_pkl.exists():
        generate_images_from_pkl(train_pkl, interface_img_dir, 'train')
    else:
        print(f"Warning: {train_pkl} not found, skipping train set")

    # Generate images for test set
    if test_pkl.exists():
        generate_images_from_pkl(test_pkl, interface_img_dir, 'test')
    else:
        print(f"Warning: {test_pkl} not found, skipping test set")

    print("\n" + "=" * 60)
    print("✓ Image generation complete!")
    print(f"Images saved to: {interface_img_dir.resolve()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
