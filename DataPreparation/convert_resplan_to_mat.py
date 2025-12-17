"""
Convert ResPlan dataset (pkl format) to Graph2plan data.mat format

This script converts a ResPlan dataset stored as a single pkl file
into the data.mat format required by Graph2plan.

Usage:
    1. First run in inspection mode to see your data structure:
       python convert_resplan_to_mat.py --inspect your_resplan.pkl

    2. Then customize the conversion function and run:
       python convert_resplan_to_mat.py --convert your_resplan.pkl output_data.mat
"""

import pickle
from sys import prefix
import numpy as np
import scipy.io as sio
from pathlib import Path
import argparse
from typing import List, Dict, Any
from collections import namedtuple

# Define the expected Graph2plan data structure
# Each data item should have these fields:
DataItem = namedtuple('DataItem', [
    'name',       # str: filename/id
    'boundary',   # array: (x,y,dir,isNew) - first two points indicate front door
    'order',      # array: room order for visualization
    'rType',      # array: room category indices
    'rBoundary',  # list of arrays: (x,y) boundary points for each room
    'gtBox',      # array: (y0,x0,y1,x1) bounding boxes from original data
    'gtBoxNew',   # array: (x0,y0,x1,y1) bounding boxes after gap filling
    'rEdge'       # array: (u,v,r) room edges with relative position
])

# Room categories mapping (from Graph2plan)
ROOM_CATEGORIES = {
    'LivingRoom': 0,
    'MasterRoom': 1,
    'Kitchen': 2,
    'Bathroom': 3,
    'DiningRoom': 4,
    'ChildRoom': 5,
    'StudyRoom': 6,
    'SecondRoom': 7,
    'GuestRoom': 8,
    'Balcony': 9,
    'Entrance': 10,
    'Storage': 11,
    'Wall-in': 12,
    'External': 13,
    'ExteriorWall': 14,
    'FrontDoor': 15,
    'InteriorWall': 16,
    'InteriorDoor': 17
}

# Edge types (relative positions)
EDGE_TYPES = {
    'left-above': 0,
    'left-below': 1,
    'left-of': 2,
    'above': 3,
    'inside': 4,
    'surrounding': 5,
    'below': 6,
    'right-of': 7,
    'right-above': 8,
    'right-below': 9
}

# ResPlan room prefix to Graph2plan category mapping
RESPLAN_TO_GRAPH2PLAN = {
    'living': 0,        # LivingRoom
    'bedroom': 1,       # MasterRoom (treat all bedrooms as this)
    'kitchen': 2,       # Kitchen
    'bathroom': 3,      # Bathroom
    'dining': 4,        # DiningRoom
    'balcony': 9,       # Balcony
    'entrance': 10,     # Entrance
    'storage': 11,      # Storage
    'front_door': 15,   # FrontDoor
    'stair': 10,        # Treat as Entrance
    # Add more mappings as needed
}


# ============================================================
# SHAPELY GEOMETRY HELPER FUNCTIONS
# ============================================================

def extract_coords_from_shapely(geometry) -> np.ndarray:
    """
    Extract coordinates from Shapely geometry objects

    Args:
        geometry: Shapely Polygon, MultiPolygon, or other geometry

    Returns:
        coords: (N x 2) array of (x, y) coordinates
    """
    try:
        from shapely.geometry import Polygon, MultiPolygon, LineString
    except ImportError:
        raise ImportError("Shapely library required. Install with: pip install shapely")

    if isinstance(geometry, Polygon):
        # Get exterior coordinates (boundary)
        coords = np.array(geometry.exterior.coords[:-1])  # Remove duplicate last point
    elif isinstance(geometry, MultiPolygon):
        # For MultiPolygon, get exterior of largest polygon
        largest = max(geometry.geoms, key=lambda p: p.area)
        coords = np.array(largest.exterior.coords[:-1])
    elif isinstance(geometry, LineString):
        coords = np.array(geometry.coords)
    else:
        # Try to get coords attribute directly
        coords = np.array(geometry.coords[:-1])

    return coords


def compute_boundary_from_rooms(room_geometries: List) -> np.ndarray:
    """
    Compute floorplan boundary from individual room geometries

    Args:
        room_geometries: List of Shapely Polygon objects for each room

    Returns:
        boundary: (N x 2) array of boundary coordinates
    """
    try:
        from shapely.ops import unary_union
    except ImportError:
        raise ImportError("Shapely library required. Install with: pip install shapely")

    # Merge all room polygons into single floorplan polygon
    floorplan = unary_union(room_geometries)

    # Extract exterior boundary
    return extract_coords_from_shapely(floorplan)


def compute_boundary_directions(coords: np.ndarray) -> np.ndarray:
    """
    Compute direction codes for boundary segments

    Direction codes:
        0 = right (→)
        1 = up (↑)
        2 = left (←)
        3 = down (↓)

    Args:
        coords: (N x 2) array of (x, y) coordinates

    Returns:
        directions: (N,) array of direction codes
    """
    n = len(coords)
    directions = np.zeros(n, dtype=int)

    for i in range(n):
        j = (i + 1) % n
        dx = coords[j, 0] - coords[i, 0]
        dy = coords[j, 1] - coords[i, 1]

        if abs(dx) > abs(dy):
            directions[i] = 0 if dx > 0 else 2
        else:
            directions[i] = 1 if dy > 0 else 3  # dy>0 => DOWN
    return directions


def add_boundary_metadata(coords: np.ndarray) -> np.ndarray:
    """
    Convert (N x 2) coordinates to Graph2plan boundary format (N x 4)

    Adds direction and isNew columns:
    - direction: 0=right, 1=up, 2=left, 3=down
    - isNew: 0 for all original corners

    Args:
        coords: (N x 2) array of (x, y) coordinates

    Returns:
        boundary: (N x 4) array of (x, y, direction, isNew)
    """
    directions = compute_boundary_directions(coords)
    is_new = np.zeros(len(coords), dtype=int)  # All original corners

    return np.column_stack([coords, directions, is_new])


def find_front_door_point(boundary: np.ndarray, room_geometries: List = None) -> np.ndarray:
    """
    Reorder boundary so first two points represent the front door

    Strategy:
    1. If room_geometries provided, find the entrance/door room
    2. Otherwise, use the bottommost-leftmost point (common convention)

    Args:
        boundary: (N x 4) array of boundary points
        room_geometries: Optional list of room geometries

    Returns:
        reordered_boundary: Boundary with front door as first two points
    """
    # Simple strategy: use bottommost point (lowest y, then leftmost x)
    coords = boundary[:, :2]

    # Find point with minimum y (bottom), break ties with minimum x (left)
    bottom_idx = np.lexsort((coords[:, 0], coords[:, 1]))[0]

    # Reorder boundary to start at this point
    reordered = np.roll(boundary, -bottom_idx, axis=0)

    return reordered


def extract_room_boundaries_from_geometries(room_geometries: List) -> List[np.ndarray]:
    """
    Extract boundary coordinates for each room from Shapely geometries

    Args:
        room_geometries: List of Shapely Polygon objects

    Returns:
        room_boundaries: List of (M x 2) arrays for each room
    """
    room_boundaries = []

    for geom in room_geometries:
        coords = extract_coords_from_shapely(geom)
        room_boundaries.append(coords)

    return room_boundaries


def compute_boxes_from_geometries(room_geometries: List) -> np.ndarray:
    """
    Compute bounding boxes from Shapely geometries

    Args:
        room_geometries: List of Shapely Polygon objects

    Returns:
        boxes: (N x 4) array in format (x0, y0, x1, y1)
    """
    boxes = []

    for geom in room_geometries:
        # Get bounding box
        minx, miny, maxx, maxy = geom.bounds
        boxes.append([minx, miny, maxx, maxy])

    return np.array(boxes)


def inspect_pkl_structure(pkl_path: str, num_samples: int = 3):
    """
    Inspect the structure of the ResPlan pkl file
    """
    print(f"\n{'='*60}")
    print(f"Inspecting: {pkl_path}")
    print(f"{'='*60}\n")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Data type: {type(data)}")

    if isinstance(data, dict):
        print(f"\nDictionary keys: {list(data.keys())}")
        for key, value in data.items():
            print(f"  {key}: {type(value)}, ", end="")
            if isinstance(value, (list, np.ndarray)):
                print(f"length={len(value)}")
            else:
                print(f"value={value}")

    # Try to access the actual floorplan data
    if isinstance(data, dict) and 'data' in data:
        floorplans = data['data']
    elif isinstance(data, list):
        floorplans = data
    else:
        floorplans = [data]

    print(f"\nTotal number of floorplans: {len(floorplans)}")
    print(f"\nInspecting first {min(num_samples, len(floorplans))} samples:")
    print("-" * 60)

    for i in range(min(num_samples, len(floorplans))):
        print(f"\nSample {i}:")
        sample = floorplans[i]
        print(f"  Type: {type(sample)}")

        if isinstance(sample, dict):
            print(f"  Keys: {list(sample.keys())}")
            for key, value in sample.items():
                print(f"    {key}: {type(value)}", end="")
                if isinstance(value, np.ndarray):
                    print(f", shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, list):
                    print(f", length={len(value)}")
                    if len(value) > 0:
                        print(f"      First element type: {type(value[0])}")
                else:
                    print(f", value={value}")

        elif hasattr(sample, '__dict__'):
            print(f"  Attributes: {list(vars(sample).keys())}")
            for key, value in vars(sample).items():
                print(f"    {key}: {type(value)}", end="")
                if isinstance(value, np.ndarray):
                    print(f", shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, list):
                    print(f", length={len(value)}")
                else:
                    print(f", value={value}")
        else:
            print(f"  Value: {sample}")

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Review the structure above")
    print("2. Update the convert_item() function to map your fields")
    print("3. Run with --convert flag to perform conversion")
    print("="*60 + "\n")


def compute_edge_relations(boxes: np.ndarray) -> np.ndarray:
    """
    Compute relative position relationships between rooms

    Args:
        boxes: (N, 4) array of boxes in format (x0, y0, x1, y1)

    Returns:
        edges: (M, 3) array of (u, v, relation_type)
    """
    edges = []
    n_rooms = len(boxes)

    for i in range(n_rooms):
        for j in range(n_rooms):
            if i == j:
                continue

            box_i = boxes[i]
            box_j = boxes[j]

            # Compute centers
            cx_i = (box_i[0] + box_i[2]) / 2
            cy_i = (box_i[1] + box_i[3]) / 2
            cx_j = (box_j[0] + box_j[2]) / 2
            cy_j = (box_j[1] + box_j[3]) / 2

            # Check if boxes overlap (are adjacent)
            overlap_x = not (box_i[2] < box_j[0] or box_j[2] < box_i[0])
            overlap_y = not (box_i[3] < box_j[1] or box_j[3] < box_i[1])

            if not (overlap_x or overlap_y):
                continue  # Not adjacent

            # Determine relative position
            dx = cx_j - cx_i
            dy = cy_j - cy_i

            # Inside/surrounding check
            if (box_j[0] >= box_i[0] and box_j[2] <= box_i[2] and
                box_j[1] >= box_i[1] and box_j[3] <= box_i[3]):
                relation = 4  # inside
            elif (box_i[0] >= box_j[0] and box_i[2] <= box_j[2] and
                  box_i[1] >= box_j[1] and box_i[3] <= box_j[3]):
                relation = 5  # surrounding
            # Directional relations
            elif abs(dx) > abs(dy):  # Horizontal relation dominant
                if dx > 0:  # j is to the right of i
                    if dy > 0:
                        relation = 8  # right-below
                    elif dy < 0:
                        relation = 9  # right-above
                    else:
                        relation = 7  # right-of
                else:  # j is to the left of i
                    if dy > 0:
                        relation = 1  # left-below
                    elif dy < 0:
                        relation = 0  # left-above
                    else:
                        relation = 2  # left-of
            else:  # Vertical relation dominant
                if dy > 0:
                    relation = 6  # below
                else:
                    relation = 3  # above

            edges.append([i, j, relation])

    return np.array(edges) if edges else np.zeros((0, 3))


# ============================================================
# RESPLAN-SPECIFIC HELPER FUNCTIONS
# ============================================================

def parse_resplan_node_name(node_name: str) -> tuple:
    """
    Parse ResPlan graph node name into (prefix, index)

    Examples:
        "living_0" → ("living", 0)
        "bathroom_2" → ("bathroom", 2)
        "front_door_0" → ("front_door", 0)

    Args:
        node_name: Node name from ResPlan graph

    Returns:
        (prefix, index): Room type prefix and room index
    """
    # Split on last underscore
    parts = node_name.rsplit('_', 1)
    if len(parts) == 2:
        prefix, idx_str = parts
        try:
            index = int(idx_str)
            return (prefix, index)
        except ValueError:
            # If second part isn't a number, treat whole thing as prefix
            return (node_name, 0)
    else:
        return (node_name, 0)


def should_include_room(prefix: str) -> bool:
    """
    Determine if a ResPlan room type should be included in Graph2plan data

    Include: living, kitchen, bedroom, bathroom, dining, balcony, entrance, storage, stair
    Exclude: wall, window, door, interior_door (these aren't spaces)

    Args:
        prefix: Room type prefix from ResPlan

    Returns:
        True if room should be included, False otherwise
    """
    exclude_list = {'wall', 'window', 'door', 'interior_door', 'exterior_wall'}
    return prefix not in exclude_list

def map_prefix_to_retype(prefix: str) -> int:
    """
    Map ResPlan room prefix to Graph2plan room category index

    Args:
        prefix: Room type prefix from ResPlan

    Returns:
        rType index for Graph2plan
    """
    return RESPLAN_TO_GRAPH2PLAN.get(prefix, ROOM_CATEGORIES["External"])


def build_room_list_from_graph(item: Dict) -> List[Dict]:
    """
    Build room list from ResPlan graph structure

    Args:
        item: ResPlan data item with 'graph' and room MultiPolygons

    Returns:
        List of room dicts with: prefix, index, polygon, rType, rBoundary, box
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("NetworkX library required. Install with: pip install networkx")

    from shapely.geometry import MultiPolygon

    graph = item['graph']
    rooms = []

    for node_name in graph.nodes():
        prefix, room_idx = parse_resplan_node_name(node_name)

        # Skip non-space nodes
        if not should_include_room(prefix):
            continue

        # Get room type category
        rtype = RESPLAN_TO_GRAPH2PLAN.get(prefix, 0)  # Default to LivingRoom if unknown

        # Get polygon from item data
        polygon = None
        if prefix in item and item[prefix] is not None:
            room_multipolygon = item[prefix]
            if isinstance(room_multipolygon, MultiPolygon):
                if room_idx < len(room_multipolygon.geoms):
                    polygon = room_multipolygon.geoms[room_idx]
                else:
                    print(f"Warning: {prefix} index {room_idx} out of range")
                    continue
            else:
                # Single polygon
                polygon = room_multipolygon
        else:
            print(f"Warning: {prefix} not found in item data")
            continue

        if polygon is None:
            continue

        # Extract boundary coords
        boundary_coords = extract_coords_from_shapely(polygon)

        # Get bounding box
        minx, miny, maxx, maxy = polygon.bounds
        box = np.array([minx, miny, maxx, maxy])

        rooms.append({
            'node_name': node_name,
            'prefix': prefix,
            'index': room_idx,
            'polygon': polygon,
            'rType': rtype,
            'rBoundary': boundary_coords,
            'box': box
        })

    return rooms


def compute_edges_from_graph(graph, node_to_index: Dict, boxes: np.ndarray) -> np.ndarray:
    """
    Compute edge relations from ResPlan graph structure

    Args:
        graph: NetworkX graph from ResPlan
        node_to_index: Mapping from node name to room index
        boxes: (N x 4) array of bounding boxes in (x0, y0, x1, y1) format

    Returns:
        edges: (M x 3) array of (u, v, relation_type)
    """
    edges = []

    for u_name, v_name in graph.edges():
        # Skip if either node not in our room list
        if u_name not in node_to_index or v_name not in node_to_index:
            continue

        u_idx = node_to_index[u_name]
        v_idx = node_to_index[v_name]

        # Compute relation from box positions
        box_u = boxes[u_idx]
        box_v = boxes[v_idx]

        # Compute centers
        cx_u = (box_u[0] + box_u[2]) / 2
        cy_u = (box_u[1] + box_u[3]) / 2
        cx_v = (box_v[0] + box_v[2]) / 2
        cy_v = (box_v[1] + box_v[3]) / 2

        dx = cx_v - cx_u
        dy = cy_v - cy_u

        # Inside/surrounding check
        if (box_v[0] >= box_u[0] and box_v[2] <= box_u[2] and
            box_v[1] >= box_u[1] and box_v[3] <= box_u[3]):
            relation = 4  # inside
        elif (box_u[0] >= box_v[0] and box_u[2] <= box_v[2] and
              box_u[1] >= box_v[1] and box_u[3] <= box_v[3]):
            relation = 5  # surrounding
        # Directional relations
        elif abs(dx) > abs(dy):  # Horizontal relation dominant
            if dx > 0:  # v is to the right of u
                if dy > 0:
                    relation = 8  # right-below
                elif dy < 0:
                    relation = 9  # right-above
                else:
                    relation = 7  # right-of
            else:  # v is to the left of u
                if dy > 0:
                    relation = 1  # left-below
                elif dy < 0:
                    relation = 0  # left-above
                else:
                    relation = 2  # left-of
        else:  # Vertical relation dominant
            if dy > 0:
                relation = 6  # below
            else:
                relation = 3  # above

        edges.append([u_idx, v_idx, relation])

    return np.array(edges) if edges else np.zeros((0, 3))


def extract_boundary_from_inner(item: Dict) -> np.ndarray:
    """
    Extract floorplan boundary from ResPlan 'inner' field

    Aligns boundary so first two points correspond to front door

    Args:
        item: ResPlan data item with 'inner' and 'front_door'

    Returns:
        boundary: (N x 4) array of (x, y, direction, isNew)
    """
    from shapely.geometry import MultiPolygon, Point

    # Get largest polygon from 'inner' MultiPolygon
    inner = item['inner']
    if isinstance(inner, MultiPolygon):
        # Get largest polygon
        largest = max(inner.geoms, key=lambda p: p.area)
    else:
        largest = inner

    # Extract boundary coordinates
    boundary_coords = extract_coords_from_shapely(largest)

    # Add direction and isNew metadata
    boundary = add_boundary_metadata(boundary_coords)

    # Try to align with front door
    if 'front_door' in item and item['front_door'] is not None:
        front_door = item['front_door']

        # Get front door centroid
        door_point = Point(front_door.centroid.x, front_door.centroid.y)

        # Find closest boundary point to door
        coords = boundary[:, :2]
        distances = np.array([
            np.sqrt((coords[i, 0] - door_point.x)**2 + (coords[i, 1] - door_point.y)**2)
            for i in range(len(coords))
        ])
        closest_idx = np.argmin(distances)

        # Reorder boundary to start at closest point
        boundary = np.roll(boundary, -closest_idx, axis=0)

        # Mark first two points as door points (isNew=1)
        boundary[0, 3] = 1
        boundary[1, 3] = 1
    else:
        # Use default ordering (bottommost-leftmost)
        boundary = find_front_door_point(boundary)

    return boundary


def convert_item(item: Any, index: int) -> Dict[str, Any]:
    """
    Convert a single ResPlan item to Graph2plan format

    Args:
        item: Single floorplan from ResPlan dataset
        index: Index for generating a unique name if needed

    Returns:
        Dictionary with Graph2plan fields

    Note:
        ResPlan structure: item['id'], item['graph'], item['inner'], item['front_door'],
        item[room_type] (MultiPolygons for each room type)
    """
    converted = {}

    # ============================================================
    # RESPLAN-SPECIFIC CONVERSION
    # ============================================================

    # Check if this is ResPlan format (has 'graph' and 'inner' fields)
    is_resplan = isinstance(item, dict) and 'graph' in item and 'inner' in item

    if is_resplan:
        # ===== RESPLAN FORMAT =====

        # 1. NAME
        converted['name'] = str(item.get('id', f'floorplan_{index:05d}'))

        # 2. BOUNDARY - Extract from 'inner' field with front door alignment
        boundary = extract_boundary_from_inner(item)
        converted['boundary'] = boundary.astype(float)

        # 3-5. BUILD ROOM LIST FROM GRAPH
        # This extracts: rType, rBoundary, boxes
        rooms = build_room_list_from_graph(item)

        if len(rooms) == 0:
            raise ValueError(f"No valid rooms found in graph for {converted['name']}")

        # Extract fields from room list
        converted['rType'] = np.array([r['rType'] for r in rooms]).astype(int)
        converted['rBoundary'] = [r['rBoundary'].astype(float) for r in rooms]

        # Boxes in (x0, y0, x1, y1) format
        boxes_new = np.array([r['box'] for r in rooms])
        converted['gtBoxNew'] = boxes_new.astype(float)

        # Convert to gtBox format: (y0, x0, y1, x1)
        converted['gtBox'] = boxes_new[:, [1, 0, 3, 2]].astype(float)

        # 6. EDGES - Compute from graph structure
        node_to_index = {r['node_name']: i for i, r in enumerate(rooms)}
        edges = compute_edges_from_graph(item['graph'], node_to_index, boxes_new)
        converted['rEdge'] = edges.astype(int)

        # 7. ORDER - Order by polygon area (larger rooms first)
        areas = np.array([r['polygon'].area for r in rooms])
        converted['order'] = np.argsort(-areas).astype(float)

    else:
        # ===== GENERIC FORMAT (FALLBACK) =====
        # For datasets that already have boundary, room_types, etc.

        # 1. NAME
        if hasattr(item, 'name'):
            converted['name'] = item.name
        elif isinstance(item, dict) and 'name' in item:
            converted['name'] = item['name']
        elif isinstance(item, dict) and 'id' in item:
            converted['name'] = str(item['id'])
        else:
            converted['name'] = f'floorplan_{index:05d}'

        # 2. BOUNDARY
        boundary = None
        if hasattr(item, 'boundary'):
            boundary = item.boundary
        elif isinstance(item, dict) and 'boundary' in item:
            boundary = item['boundary']
        elif hasattr(item, 'geometry'):
            boundary_coords = extract_coords_from_shapely(item.geometry)
            boundary = add_boundary_metadata(boundary_coords)
        elif isinstance(item, dict) and 'geometry' in item:
            boundary_coords = extract_coords_from_shapely(item['geometry'])
            boundary = add_boundary_metadata(boundary_coords)
        else:
            raise ValueError("Boundary not found and not in ResPlan format")

        if not isinstance(boundary, np.ndarray):
            boundary = np.array(boundary)
        if boundary.shape[1] == 2:
            boundary = add_boundary_metadata(boundary)
        boundary = find_front_door_point(boundary)
        converted['boundary'] = boundary.astype(float)

        # 3. ROOM TYPES
        if hasattr(item, 'rType'):
            r_type = item.rType
        elif isinstance(item, dict) and 'room_types' in item:
            r_type = item['room_types']
        elif isinstance(item, dict) and 'rType' in item:
            r_type = item['rType']
        else:
            raise ValueError("Room types not found")
        converted['rType'] = np.array(r_type).astype(int)

        # 4. ROOM BOUNDARIES
        r_boundary = None
        if hasattr(item, 'rBoundary'):
            r_boundary = item.rBoundary
        elif isinstance(item, dict) and 'room_boundaries' in item:
            r_boundary = item['room_boundaries']
        elif isinstance(item, dict) and 'rBoundary' in item:
            r_boundary = item['rBoundary']
        else:
            r_boundary = []

        if r_boundary:
            converted['rBoundary'] = [np.array(rb).astype(float) for rb in r_boundary]
        else:
            converted['rBoundary'] = []

        # 5. BOUNDING BOXES
        if hasattr(item, 'gtBox'):
            gt_box = item.gtBox
        elif isinstance(item, dict) and 'boxes' in item:
            boxes = np.array(item['boxes'])
            gt_box = boxes[:, [1, 0, 3, 2]]  # Convert to (y0, x0, y1, x1)
        elif isinstance(item, dict) and 'gtBox' in item:
            gt_box = item['gtBox']
        else:
            raise ValueError("Bounding boxes not found")

        converted['gtBox'] = np.array(gt_box).astype(float)

        if hasattr(item, 'gtBoxNew'):
            gt_box_new = item.gtBoxNew
        elif isinstance(item, dict) and 'gtBoxNew' in item:
            gt_box_new = item['gtBoxNew']
        else:
            gt_box_new = gt_box[:, [1, 0, 3, 2]]  # Convert to (x0, y0, x1, y1)
        converted['gtBoxNew'] = np.array(gt_box_new).astype(float)

        # 6. EDGES
        if hasattr(item, 'rEdge'):
            r_edge = item.rEdge
        elif isinstance(item, dict) and 'edges' in item:
            r_edge = item['edges']
        elif isinstance(item, dict) and 'rEdge' in item:
            r_edge = item['rEdge']
        else:
            r_edge = compute_edge_relations(converted['gtBoxNew'])
        converted['rEdge'] = np.array(r_edge).astype(int)

        # 7. ORDER
        if hasattr(item, 'order'):
            order = item.order
        elif isinstance(item, dict) and 'order' in item:
            order = item['order']
        else:
            boxes = converted['gtBoxNew']
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            order = np.argsort(-areas)
        converted['order'] = np.array(order).astype(float)

    return converted


def convert_resplan_to_mat(pkl_path: str, output_path: str):
    """
    Convert ResPlan pkl file to Graph2plan data.mat format
    """
    print(f"\nLoading ResPlan data from: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Extract floorplan list
    if isinstance(data, dict) and 'data' in data:
        floorplans = data['data']
    elif isinstance(data, list):
        floorplans = data
    else:
        floorplans = [data]

    print(f"Found {len(floorplans)} floorplans")
    print(f"\nConverting to Graph2plan format...")

    converted_data = []
    failed_indices = []

    for i, item in enumerate(floorplans):
        try:
            converted = convert_item(item, i)

            # Create structured array for MATLAB
            dt = np.dtype([
                ('name', 'O'),
                ('boundary', 'O'),
                ('order', 'O'),
                ('rType', 'O'),
                ('rBoundary', 'O'),
                ('gtBox', 'O'),
                ('gtBoxNew', 'O'),
                ('rEdge', 'O')
            ])

            structured_item = np.empty(1, dtype=dt)

            structured_item['name'][0] = converted['name']
            structured_item['boundary'][0] = converted['boundary']
            structured_item['order'][0] = converted['order']
            structured_item['rType'][0] = converted['rType']
            structured_item['rBoundary'][0] = converted['rBoundary']
            structured_item['gtBox'][0] = converted['gtBox']
            structured_item['gtBoxNew'][0] = converted['gtBoxNew']
            structured_item['rEdge'][0] = converted['rEdge']

            converted_data.append(structured_item[0])


            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(floorplans)}")

        except Exception as e:
            print(f"  Failed to convert item {i}: {e}")
            failed_indices.append(i)

    print(f"\nSuccessfully converted: {len(converted_data)}/{len(floorplans)}")
    if failed_indices:
        print(f"Failed indices: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")

    # Save as .mat file
    print(f"\nSaving to: {output_path}")
    converted_array = np.array(converted_data, dtype=dt)
    sio.savemat(output_path, {'data': converted_array},
                format='5', oned_as='row')

    print(f"✓ Conversion complete!")
    print(f"\nNext steps:")
    print(f"1. Update DataPreparation/config.py to point to {output_path}")
    print(f"2. Run the data preparation scripts 1-6 in order")


def main():
    parser = argparse.ArgumentParser(
        description='Convert ResPlan pkl to Graph2plan data.mat format'
    )
    parser.add_argument('input_pkl', type=str,
                       help='Path to ResPlan pkl file')
    parser.add_argument('--inspect', action='store_true',
                       help='Inspect the structure of the pkl file')
    parser.add_argument('--convert', action='store_true',
                       help='Convert the pkl file to data.mat')
    parser.add_argument('--output', type=str, default='data.mat',
                       help='Output path for data.mat (default: data.mat)')
    parser.add_argument('--samples', type=int, default=3,
                       help='Number of samples to inspect (default: 3)')

    args = parser.parse_args()

    if not Path(args.input_pkl).exists():
        print(f"Error: File not found: {args.input_pkl}")
        return

    if args.inspect:
        inspect_pkl_structure(args.input_pkl, args.samples)
    elif args.convert:
        convert_resplan_to_mat(args.input_pkl, args.output)
    else:
        print("Please specify either --inspect or --convert")
        parser.print_help()


if __name__ == '__main__':
    main()
