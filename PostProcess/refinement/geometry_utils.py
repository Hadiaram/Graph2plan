"""
Geometric utility functions for post-processing refinement.

This module provides basic geometric operations used throughout
the refinement pipeline.
"""

import numpy as np
from typing import Tuple, List, Optional


class Box:
    """
    Represents a rectangular bounding box.

    Attributes:
        x1, y1: Top-left corner coordinates
        x2, y2: Bottom-right corner coordinates
    """

    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x1, y1, x2, y2]"""
        return np.array([self.x1, self.y1, self.x2, self.y2])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Box':
        """Create Box from numpy array [x1, y1, x2, y2]"""
        return cls(arr[0], arr[1], arr[2], arr[3])

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def __repr__(self):
        return f"Box({self.x1:.2f}, {self.y1:.2f}, {self.x2:.2f}, {self.y2:.2f})"


class BoundarySegment:
    """
    Represents a line segment on the boundary.

    Attributes:
        x1, y1: Start point
        x2, y2: End point
        orientation: 'horizontal' or 'vertical'
    """

    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)

        # Determine orientation
        if abs(y2 - y1) < 1e-6:
            self.orientation = 'horizontal'
        elif abs(x2 - x1) < 1e-6:
            self.orientation = 'vertical'
        else:
            self.orientation = 'diagonal'

    @property
    def length(self) -> float:
        return np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)

    def __repr__(self):
        return f"Segment({self.x1:.2f}, {self.y1:.2f}) -> ({self.x2:.2f}, {self.y2:.2f}) [{self.orientation}]"


def extract_boundary_segments(boundary: np.ndarray) -> Tuple[List[BoundarySegment], List[BoundarySegment]]:
    """
    Extract horizontal and vertical segments from boundary polygon.

    Args:
        boundary: Nx2 or Nx3+ array of boundary vertices [[x, y], ...]
                 If Nx4, assumes format [x, y, orientation, isNew]

    Returns:
        h_segments: List of horizontal boundary segments
        v_segments: List of vertical boundary segments
    """
    # Handle different boundary formats
    if boundary.shape[1] >= 4:
        # Format: [x, y, orientation, isNew, ...]
        # Filter out "new" vertices if that column exists
        is_new = boundary[:, 3]
        boundary_points = boundary[is_new == 0, :2]
    else:
        boundary_points = boundary[:, :2]

    h_segments = []
    v_segments = []

    # Create segments from consecutive vertices
    n_points = len(boundary_points)
    for i in range(n_points):
        p1 = boundary_points[i]
        p2 = boundary_points[(i + 1) % n_points]  # Wrap around

        seg = BoundarySegment(p1[0], p1[1], p2[0], p2[1])

        if seg.orientation == 'horizontal':
            h_segments.append(seg)
        elif seg.orientation == 'vertical':
            v_segments.append(seg)
        # Skip diagonal segments for now

    return h_segments, v_segments


def distance_point_to_segment(px: float, py: float,
                              seg: BoundarySegment) -> float:
    """
    Calculate minimum distance from a point to a line segment.

    Args:
        px, py: Point coordinates
        seg: Line segment

    Returns:
        Minimum distance from point to segment
    """
    x1, y1 = seg.x1, seg.y1
    x2, y2 = seg.x2, seg.y2

    # Vector from segment start to point
    dx = px - x1
    dy = py - y1

    # Segment vector
    sx = x2 - x1
    sy = y2 - y1

    # Segment length squared
    seg_len_sq = sx*sx + sy*sy

    if seg_len_sq < 1e-10:
        # Degenerate segment (point)
        return np.sqrt(dx*dx + dy*dy)

    # Project point onto segment line
    t = (dx * sx + dy * sy) / seg_len_sq

    # Clamp to segment bounds [0, 1]
    t = max(0.0, min(1.0, t))

    # Closest point on segment
    closest_x = x1 + t * sx
    closest_y = y1 + t * sy

    # Distance to closest point
    dist_x = px - closest_x
    dist_y = py - closest_y

    return np.sqrt(dist_x*dist_x + dist_y*dist_y)


def point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.

    Args:
        x, y: Point coordinates
        polygon: Nx2 array of polygon vertices

    Returns:
        True if point is inside polygon, False otherwise
    """
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]

        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside

        p1x, p1y = p2x, p2y

    return inside


def clip_box_to_boundary(box: Box, boundary: np.ndarray) -> Box:
    """
    Clip a box to fit within boundary polygon (simple axis-aligned clipping).

    Args:
        box: Box to clip
        boundary: Boundary polygon

    Returns:
        Clipped box
    """
    # Get boundary extents
    if boundary.shape[1] >= 4:
        is_new = boundary[:, 3]
        boundary_points = boundary[is_new == 0, :2]
    else:
        boundary_points = boundary[:, :2]

    x_min = np.min(boundary_points[:, 0])
    x_max = np.max(boundary_points[:, 0])
    y_min = np.min(boundary_points[:, 1])
    y_max = np.max(boundary_points[:, 1])

    # Clip box to boundary extents
    clipped = Box(
        max(box.x1, x_min),
        max(box.y1, y_min),
        min(box.x2, x_max),
        min(box.y2, y_max)
    )

    # Ensure valid box (x2 > x1, y2 > y1)
    if clipped.x2 <= clipped.x1:
        clipped.x2 = clipped.x1 + 1
    if clipped.y2 <= clipped.y1:
        clipped.y2 = clipped.y1 + 1

    return clipped


if __name__ == "__main__":
    # Quick tests
    print("Testing geometry utilities...")

    # Test Box
    box = Box(10, 20, 50, 60)
    print(f"Box: {box}")
    print(f"  Width: {box.width}, Height: {box.height}")
    print(f"  Area: {box.area}")
    print(f"  Center: {box.center}")

    # Test BoundarySegment
    seg_h = BoundarySegment(0, 10, 100, 10)
    seg_v = BoundarySegment(50, 0, 50, 100)
    print(f"\nHorizontal segment: {seg_h}")
    print(f"Vertical segment: {seg_v}")

    # Test distance
    dist = distance_point_to_segment(50, 50, seg_h)
    print(f"\nDistance from (50, 50) to horizontal segment: {dist:.2f}")

    # Test point in polygon
    polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    print(f"\nPoint (50, 50) in polygon: {point_in_polygon(50, 50, polygon)}")
    print(f"Point (150, 50) in polygon: {point_in_polygon(150, 50, polygon)}")

    print("\n✓ Geometry utilities working!")
