# load_resplan_graphs.py
# Load graphs from ResPlan PKL files for edge prediction training

import os
import pickle
import re
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_undirected, remove_self_loops
from shapely.geometry import MultiPolygon

# =========================
# Canonical type mapping (matching train_semantic_edge_model_residential.py)
# =========================
CANON_MAP = [
    (r"(corridor|passage|hall|foyer|lobby)", "circulation"),
    (r"\bbed(room)?\b", "bedroom"),
    (r"(bath|toilet|wc|powder|half bath)", "bath"),
    (r"(kitchen|pantry)", "kitchen"),
    (r"(living|lounge)", "living"),
    (r"(laundry|utility|wash|washer|dryer)", "laundry"),
    (r"(dressing|wardrobe|closet|walk-?in)", "closet"),
    (r"(office|faclty off|off dir)", "office"),
    (r"(clinic|ward|osce)", "clinic"),
    (r"(storage|store|riser|tl closet)", "storage"),
    (r"(mechanical|mech|pump|electrical|control electrical|grey water)", "mech"),
    (r"\bstair\b", "stair"),
    (r"(elevator|lift)", "elevator"),
    (r"(balcony|terrace)", "balcony"),
    (r"(entrance|entry|vestibule)", "entrance"),
]

def canonical_type_from_text(text):
    """Map room name/type to canonical category."""
    txt = text.lower().strip()
    txt = re.sub(r"[\s\-_]*\d+\s*$", "", txt)  # Remove trailing numbers
    txt = txt.replace("living room", "living").replace("master bedroom", "bedroom")
    for pat, lab in CANON_MAP:
        if re.search(pat, txt):
            return lab
    return "other"


def get_polygon_centroid(polygon):
    """Get centroid from Shapely polygon/multipolygon in millimeters."""
    try:
        if isinstance(polygon, MultiPolygon):
            # Use the largest polygon if it's a multipolygon
            polygon = max(polygon.geoms, key=lambda p: p.area)

        centroid = polygon.centroid
        # ResPlan coordinates are already in millimeters (based on conversion code)
        return [centroid.x, centroid.y, 0.0]
    except Exception as e:
        print(f"Warning: Could not compute centroid: {e}")
        return [0.0, 0.0, 0.0]


def load_resplan_graphs(folder, type_to_id, file_limit=None):
    """
    Load graphs from ResPlan PKL files.

    Args:
        folder: Path to directory containing ResPlan PKL files
        type_to_id: Dict mapping canonical room types to integer IDs (updated in-place)
        file_limit: Optional limit on number of files to load (for testing)

    Returns:
        List of PyTorch Geometric Data objects
    """
    graphs = []
    files_processed = 0

    pkl_files = [f for f in os.listdir(folder) if f.endswith('.pkl')]
    print(f"Found {len(pkl_files)} PKL files in {folder}")

    for fn in pkl_files:
        if file_limit and files_processed >= file_limit:
            print(f"Reached file limit of {file_limit}")
            break

        path = os.path.join(folder, fn)

        try:
            with open(path, 'rb') as f:
                resplan_data = pickle.load(f)
        except Exception as e:
            print(f"❌ Skip {fn}: Failed to load - {e}")
            continue

        # Extract the graph
        if 'graph' not in resplan_data:
            print(f"❌ Skip {fn}: No 'graph' field")
            continue

        G = resplan_data['graph']

        if len(G.nodes()) == 0:
            print(f"⚠️ Skip {fn}: Empty graph")
            continue

        # Build node features
        # ResPlan graph nodes are labeled by room type (e.g., 'living_0', 'bedroom_1')
        fallback_coords = 0

        for node_id in G.nodes():
            # Get room type from node ID
            # ResPlan nodes are typically named like: 'living_0', 'bedroom_1', 'bathroom_0'
            node_name = str(node_id)

            # Extract room type prefix (before underscore and number)
            room_type = re.sub(r"_\d+$", "", node_name)  # Remove trailing _N

            # Canonicalize the room type
            canonical = canonical_type_from_text(room_type)

            # Add to vocabulary if new
            if canonical not in type_to_id:
                type_to_id[canonical] = len(type_to_id)

            type_id = float(type_to_id[canonical])

            # Get centroid coordinates from corresponding polygon
            coords_m = None

            # Try to get polygon from ResPlan data
            # ResPlan stores room geometries as: data['living'], data['bedroom'], etc.
            if room_type in resplan_data:
                polygon = resplan_data[room_type]
                if polygon and not polygon.is_empty:
                    coords_m = get_polygon_centroid(polygon)

            # Fallback: try to get from node attributes if available
            if coords_m is None and 'centroid' in G.nodes[node_id]:
                c = G.nodes[node_id]['centroid']
                coords_m = [c.x, c.y, 0.0]

            # Last fallback: use zeros
            if coords_m is None:
                coords_m = [0.0, 0.0, 0.0]
                fallback_coords += 1

            # Store feature as [type_id, x, y, z]
            G.nodes[node_id]['feat'] = [type_id] + coords_m
            G.nodes[node_id]['name'] = room_type  # Store for debugging

        # Convert to PyTorch Geometric Data
        try:
            data = from_networkx(G, group_node_attrs=["feat"])
        except KeyError:
            missing = [n for n in G.nodes() if 'feat' not in G.nodes[n]]
            print(f"⚠️ {fn}: nodes missing 'feat' ({len(missing)}); skipping.")
            continue

        # Normalize features across PyG versions
        if hasattr(data, 'x') and data.x is not None:
            data.x = data.x.to(torch.float32)
        elif hasattr(data, 'feat'):
            data.x = torch.as_tensor(data.feat, dtype=torch.float32)
            try:
                delattr(data, 'feat')
            except Exception:
                pass
        else:
            # Last-resort: rebuild from G
            feats = [G.nodes[n]['feat'] for n in G.nodes()]
            data.x = torch.tensor(feats, dtype=torch.float32)

        # Undirect edges & remove self-loops
        if hasattr(data, 'edge_index'):
            ei, _ = remove_self_loops(data.edge_index)
            data.edge_index = to_undirected(ei)
        else:
            # If no edges, create empty edge_index
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Only add graphs with edges
        if data.edge_index.size(1) > 0:
            graphs.append(data)
            files_processed += 1
            if fallback_coords > 0:
                print(f"✅ Loaded {fn} ({fallback_coords} fallback coords)")
            else:
                print(f"✅ Loaded {fn}")
        else:
            print(f"⚠️ Skipped {fn}: no edges after processing")

    print(f"\n📦 Loaded {len(graphs)} graphs from {folder}")
    return graphs


def load_resplan_from_converted_pkl(pkl_path, type_to_id):
    """
    Alternative loader: Load from the converted data_train_converted.pkl format.
    This is the format created by convert_resplan_to_mat.py and already used in Graph2plan.

    Args:
        pkl_path: Path to data_train_converted.pkl or data_test_converted.pkl
        type_to_id: Dict mapping canonical room types to integer IDs (updated in-place)

    Returns:
        List of PyTorch Geometric Data objects
    """
    print(f"Loading from converted PKL: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        converted_data = pickle.load(f)

    # The converted format has: {'data': [list of floor plans], 'name': [list of IDs]}
    if isinstance(converted_data, dict):
        floor_plans = converted_data.get('data', converted_data)
    else:
        floor_plans = converted_data

    # Handle scipy.io.loadmat squeeze_me=True squeezing arrays
    if not isinstance(floor_plans, (list, np.ndarray)):
        floor_plans = [floor_plans]
    elif isinstance(floor_plans, np.ndarray):
        if floor_plans.ndim == 0:
            floor_plans = [floor_plans.item()]
        elif floor_plans.ndim == 1:
            floor_plans = list(floor_plans)
        else:
            floor_plans = list(floor_plans.flat)

    graphs = []

    # Helper function to handle both dict and mat_struct object access
    def get_attr(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        else:
            return getattr(obj, key, None)

    for idx, fp in enumerate(floor_plans):
        # Each fp has: box (room boxes), edge (edges) - can be dict or mat_struct
        try:
            # Get room types from box data (last column is room type)
            boxes = get_attr(fp, 'box')  # Shape: (N, 5) - [x1, y1, x2, y2, room_type]
            edges = get_attr(fp, 'edge')  # Shape: (E, 3) - [room1, room2, edge_type]

            if boxes is None:
                print(f"⚠️ Skip floor plan {idx}: no box data")
                continue

            # Convert to numpy array if needed
            if not isinstance(boxes, np.ndarray):
                boxes = np.array(boxes)
            if edges is not None and not isinstance(edges, np.ndarray):
                edges = np.array(edges)

            num_rooms = boxes.shape[0]

            # Build node features
            node_features = []
            for i in range(num_rooms):
                room_type_id = int(boxes[i, 4])

                # Map back to canonical name (this is already in Graph2plan format 0-17)
                # Create a consistent mapping to sequential indices for embedding layer
                canonical = f"type_{room_type_id}"

                if canonical not in type_to_id:
                    # Assign sequential index starting from 0
                    type_to_id[canonical] = len(type_to_id)

                # Use the mapped sequential index, not the raw room_type_id
                mapped_type_id = type_to_id[canonical]

                # Get centroid from bounding box
                x1, y1, x2, y2 = boxes[i, :4]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # Feature: [mapped_type_id, cx, cy, 0]
                node_features.append([float(mapped_type_id), cx, cy, 0.0])

            # Build edge index
            edge_index = []
            if edges is not None and edges.shape[0] > 0:
                for e in edges:
                    src, dst = int(e[0]), int(e[1])
                    edge_index.append([src, dst])
                    edge_index.append([dst, src])  # Make undirected

            # Create PyG Data object
            x = torch.tensor(node_features, dtype=torch.float32)

            if len(edge_index) > 0:
                edge_idx = torch.tensor(edge_index, dtype=torch.long).T
                data = Data(x=x, edge_index=edge_idx, num_nodes=num_rooms)
                graphs.append(data)

                if idx % 1000 == 0:
                    print(f"Processed {idx}/{len(floor_plans)} floor plans...")
            else:
                print(f"⚠️ Skip floor plan {idx}: no edges")

        except Exception as e:
            print(f"❌ Skip floor plan {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n📦 Loaded {len(graphs)} graphs from converted PKL")
    return graphs


# =========================
# Test/Example Usage
# =========================
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("ResPlan Graph Loader - Test")
    print("=" * 60)

    # Example 1: Load from individual ResPlan PKL files
    resplan_folder = r"C:\Users\hmbashir\source\ResPlan_Dataset\plans_split"

    if os.path.isdir(resplan_folder):
        print(f"\n1. Testing load from ResPlan PKL files...")
        print(f"   Folder: {resplan_folder}")

        type_to_id = {}
        graphs = load_resplan_graphs(resplan_folder, type_to_id, file_limit=10)

        print(f"\n✅ Loaded {len(graphs)} graphs")
        print(f"✅ Found {len(type_to_id)} room types: {list(type_to_id.keys())}")

        if len(graphs) > 0:
            print(f"\nExample graph:")
            print(f"  Nodes: {graphs[0].num_nodes}")
            print(f"  Edges: {graphs[0].edge_index.size(1)}")
            print(f"  Features shape: {graphs[0].x.shape}")
            print(f"  First 3 features:\n{graphs[0].x[:3]}")
    else:
        print(f"⚠️ Folder not found: {resplan_folder}")

    # Example 2: Load from converted PKL (Graph2plan format)
    converted_pkl = r"C:\Users\hmbashir\source\Graph2plan\Interface\static\Data\data_train_converted.pkl"

    if os.path.isfile(converted_pkl):
        print(f"\n2. Testing load from converted PKL...")
        print(f"   File: {converted_pkl}")

        type_to_id_2 = {}
        graphs_2 = load_resplan_from_converted_pkl(converted_pkl, type_to_id_2)

        print(f"\n✅ Loaded {len(graphs_2)} graphs")
        print(f"✅ Found {len(type_to_id_2)} room types")

        if len(graphs_2) > 0:
            print(f"\nExample graph:")
            print(f"  Nodes: {graphs_2[0].num_nodes}")
            print(f"  Edges: {graphs_2[0].edge_index.size(1)}")
            print(f"  Features shape: {graphs_2[0].x.shape}")
    else:
        print(f"⚠️ File not found: {converted_pkl}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
