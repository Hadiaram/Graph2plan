"""
Visualization tools for post-processing refinement.

Provides functions to visualize boxes, boundaries, and refinement steps.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, List, Dict
from .geometry_utils import Box


# Room type colors (matching Graph2plan convention)
ROOM_COLORS = {
    0: '#FFB6C1',  # Living Room - Light Pink
    1: '#87CEEB',  # Bedroom - Sky Blue
    2: '#98FB98',  # Kitchen - Pale Green
    3: '#DDA0DD',  # Bathroom - Plum
    4: '#F0E68C',  # Balcony - Khaki
    5: '#FFD700',  # Entrance - Gold
    6: '#FFA07A',  # Dining Room - Light Salmon
    7: '#E6E6FA',  # Closet - Lavender
    8: '#F5DEB3',  # Storage - Wheat
    9: '#D3D3D3',  # Other - Light Gray
}

ROOM_NAMES = {
    0: 'Living Room',
    1: 'Bedroom',
    2: 'Kitchen',
    3: 'Bathroom',
    4: 'Balcony',
    5: 'Entrance',
    6: 'Dining Room',
    7: 'Closet',
    8: 'Storage',
    9: 'Other',
}


def plot_boundary(ax, boundary: np.ndarray, color='black', linewidth=2, label='Boundary'):
    """Plot boundary polygon."""
    # Handle different boundary formats
    if boundary.shape[1] >= 4:
        is_new = boundary[:, 3]
        boundary_points = boundary[is_new == 0, :2]
    else:
        boundary_points = boundary[:, :2]

    # Close the polygon
    boundary_closed = np.vstack([boundary_points, boundary_points[0]])

    ax.plot(boundary_closed[:, 0], boundary_closed[:, 1],
            color=color, linewidth=linewidth, label=label, zorder=10)


def plot_box(ax, box: np.ndarray, room_type: Optional[int] = None,
             color: Optional[str] = None, alpha: float = 0.3,
             edgecolor: str = 'black', linewidth: float = 1.5,
             label: Optional[str] = None):
    """
    Plot a single box.

    Args:
        ax: Matplotlib axis
        box: Box as array [x1, y1, x2, y2] or Box object
        room_type: Optional room type index for color
        color: Optional explicit color (overrides room_type)
        alpha: Transparency (0-1)
        edgecolor: Edge color
        linewidth: Edge line width
        label: Optional label
    """
    if isinstance(box, Box):
        x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
    else:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

    # Determine fill color
    if color is None:
        if room_type is not None:
            color = ROOM_COLORS.get(room_type, ROOM_COLORS[9])
        else:
            color = '#CCCCCC'

    # Create rectangle
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=color,
        alpha=alpha,
        label=label
    )

    ax.add_patch(rect)

    # Add room type label if provided
    if room_type is not None:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        room_name = ROOM_NAMES.get(room_type, f'Type {room_type}')
        ax.text(cx, cy, room_name, ha='center', va='center',
                fontsize=8, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))


def plot_boxes(ax, boxes: np.ndarray, room_types: Optional[np.ndarray] = None,
               alpha: float = 0.3, title: Optional[str] = None):
    """
    Plot multiple boxes.

    Args:
        ax: Matplotlib axis
        boxes: Nx4 array of boxes [[x1, y1, x2, y2], ...]
        room_types: Optional array of room type indices
        alpha: Transparency
        title: Optional subplot title
    """
    for i, box in enumerate(boxes):
        room_type = room_types[i] if room_types is not None else None
        plot_box(ax, box, room_type=room_type, alpha=alpha)

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


def visualize_boundary_alignment(original_boxes: np.ndarray,
                                aligned_boxes: np.ndarray,
                                boundary: np.ndarray,
                                room_types: Optional[np.ndarray] = None,
                                updated_edges: Optional[List[Dict[str, bool]]] = None,
                                title: str = "Boundary Alignment",
                                save_path: Optional[str] = None):
    """
    Visualize before and after boundary alignment.

    Args:
        original_boxes: Original boxes before alignment
        aligned_boxes: Boxes after alignment
        boundary: Boundary polygon
        room_types: Optional room type indices
        updated_edges: Optional list of updated edge information
        title: Figure title
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Plot original
    plot_boundary(ax1, boundary)
    plot_boxes(ax1, original_boxes, room_types=room_types, title="Before Alignment")

    # Plot aligned
    plot_boundary(ax2, boundary)
    plot_boxes(ax2, aligned_boxes, room_types=room_types, title="After Alignment")

    # Highlight updated edges
    if updated_edges is not None:
        for i, (orig_box, aligned_box, edges) in enumerate(zip(original_boxes, aligned_boxes, updated_edges)):
            if any(edges.values()):
                # Draw arrows showing the snap
                x1_o, y1_o, x2_o, y2_o = orig_box
                x1_a, y1_a, x2_a, y2_a = aligned_box

                if edges.get('left', False) and abs(x1_o - x1_a) > 0.1:
                    ax2.annotate('', xy=(x1_a, (y1_a + y2_a)/2), xytext=(x1_o, (y1_o + y2_o)/2),
                               arrowprops=dict(arrowstyle='->', color='red', lw=2))

                if edges.get('right', False) and abs(x2_o - x2_a) > 0.1:
                    ax2.annotate('', xy=(x2_a, (y1_a + y2_a)/2), xytext=(x2_o, (y1_o + y2_o)/2),
                               arrowprops=dict(arrowstyle='->', color='red', lw=2))

                if edges.get('top', False) and abs(y1_o - y1_a) > 0.1:
                    ax2.annotate('', xy=((x1_a + x2_a)/2, y1_a), xytext=((x1_o + x2_o)/2, y1_o),
                               arrowprops=dict(arrowstyle='->', color='red', lw=2))

                if edges.get('bottom', False) and abs(y2_o - y2_a) > 0.1:
                    ax2.annotate('', xy=((x1_a + x2_a)/2, y2_a), xytext=((x1_o + x2_o)/2, y2_o),
                               arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Set equal scaling
    for ax in [ax1, ax2]:
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


def visualize_refinement_pipeline(boxes_dict: Dict[str, np.ndarray],
                                  boundary: np.ndarray,
                                  room_types: Optional[np.ndarray] = None,
                                  title: str = "Refinement Pipeline",
                                  save_path: Optional[str] = None):
    """
    Visualize multiple refinement steps in a grid.

    Args:
        boxes_dict: Dictionary of {step_name: boxes_array}
                   e.g., {'Original': ..., 'Boundary Aligned': ..., 'Neighbor Aligned': ...}
        boundary: Boundary polygon
        room_types: Optional room type indices
        title: Figure title
        save_path: Optional path to save figure
    """
    n_steps = len(boxes_dict)
    ncols = min(3, n_steps)
    nrows = (n_steps + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 7*nrows))

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (step_name, boxes) in enumerate(boxes_dict.items()):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        plot_boundary(ax, boundary)
        plot_boxes(ax, boxes, room_types=room_types, title=step_name)

    # Hide unused subplots
    for idx in range(n_steps, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


if __name__ == "__main__":
    print("Testing visualization...")

    # Create test data
    boundary = np.array([
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100]
    ])

    boxes = np.array([
        [5, 10, 40, 45],    # Living room (close to left wall)
        [45, 10, 75, 45],   # Bedroom (middle)
        [80, 10, 97, 45],   # Kitchen (close to right wall)
        [5, 55, 40, 95],    # Bathroom (close to left wall)
        [45, 55, 97, 95],   # Balcony (wide)
    ])

    room_types = np.array([0, 1, 2, 3, 4])

    # Simulate alignment (just snap boxes close to walls)
    aligned_boxes = boxes.copy()
    aligned_boxes[0, 0] = 0  # Snap living room left edge
    aligned_boxes[2, 2] = 100  # Snap kitchen right edge
    aligned_boxes[3, 0] = 0  # Snap bathroom left edge

    print("\nVisualizing boundary alignment...")
    visualize_boundary_alignment(
        boxes, aligned_boxes, boundary,
        room_types=room_types,
        title="Test Boundary Alignment"
    )

    print("\n✓ Visualization test complete!")
