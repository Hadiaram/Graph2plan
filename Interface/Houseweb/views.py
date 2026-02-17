from django.shortcuts import render # type: ignore
from django.http import HttpResponse, JsonResponse # type: ignore
from django.conf import settings # type: ignore
import json
import io
import os
import random
import model.test as mltest
import model.utils as mdul
from model.floorplan import *
import retrieval.retrieval as rt
import time
import pickle
import scipy.io as sio
import numpy as np
from model.decorate import *
import math
import pandas as pd # type: ignore
import warnings
import sys
from pathlib import Path
try:
    from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box, Point as ShapelyPoint
    HAS_SHAPELY_VIEWS = True
except ImportError:
    HAS_SHAPELY_VIEWS = False

# Try to import DXF export
HAS_DXF_EXPORT = False
try:
    from Houseweb.dxf_export import save_floorplan_dxf
    HAS_DXF_EXPORT = True
    print("[Init] DXF export module loaded successfully ✓")
except ImportError as e:
    print(f"[Init] DXF export not available: {e}")
    warnings.warn("DXF export not available. Install ezdxf: pip install ezdxf", UserWarning)

# Add parent directory to path for PostProcess imports
GRAPH2PLAN_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(GRAPH2PLAN_ROOT))

# Try to import MATLAB - make it optional
HAS_MATLAB = False
try:
    import matlab.engine #type: ignore
    HAS_MATLAB = True
except ImportError:
    warnings.warn("MATLAB engine not available. Some features may be limited.", UserWarning)

# Try to import Python refinement
HAS_PYTHON_REFINEMENT = False
try:
    from PostProcess.refinement.integration import align_fp_python
    HAS_PYTHON_REFINEMENT = True
    print("[Init] Python geometric refinement loaded successfully ✓")
except ImportError as e:
    print(f"[Init] Python refinement not available: {e}")
    warnings.warn("Python refinement not available. Will use MATLAB or basic fallback.", UserWarning)

# ---------------------------------------------------------------------------
# Log capture helpers
# ---------------------------------------------------------------------------

class _TeeBuffer:
    """Write to both a StringIO buffer and the original stdout simultaneously."""
    def __init__(self, real_stdout):
        self._buf = io.StringIO()
        self._real = real_stdout

    def write(self, data):
        self._buf.write(data)
        self._real.write(data)

    def flush(self):
        self._buf.flush()
        self._real.flush()

    def getvalue(self):
        return self._buf.getvalue()


# Accumulates logs across all refinement passes so the full session can be
# downloaded. Resets automatically when the pass cycle restarts at Pass 1.
_last_refinement_log: dict = {"userRoomID": None, "text": ""}

# ---------------------------------------------------------------------------

global test_data, test_data_topk, testNameList, trainNameList
global train_data, trainTF, train_data_eNum, train_data_rNum
global engview, model
global tf_train, centroids, clusters
global boxes_pred, indxlist

# Initialize module-level variables
boxes_pred = None
indxlist = None

# DXF Export Configuration
ENABLE_AUTO_DXF_EXPORT = False  # Set to True to auto-save DXF on every save (disabled by default)
DXF_SCALE = 1.0  # Scale factor (1.0 = pixels, 0.01 = cm, 0.0254 = inches)
DXF_WALL_THICKNESS = 3.0  # Wall thickness in drawing units

# DXF Save Path
DXF_SAVE_PATH = r"C:\Users\hmbashir\source\DXF Floor Plans"  # Change this to your desired path


def _clip_box_to_polygon(x1, y1, x2, y2, xmin, xmax, ymin, ymax, margin, boundary_shape):
    """
    Clip a box to the boundary bounding box, then translate it so its
    center is inside the actual boundary polygon (for non-rectangular boundaries).
    Returns (x1, y1, x2, y2).
    """
    # Step 1: bounding-box clamp
    x1 = max(xmin + margin, min(x1, xmax - margin))
    x2 = max(xmin + margin, min(x2, xmax - margin))
    y1 = max(ymin + margin, min(y1, ymax - margin))
    y2 = max(ymin + margin, min(y2, ymax - margin))
    if x2 <= x1: x2 = x1 + 10
    if y2 <= y1: y2 = y1 + 10

    # Step 2: polygon-aware translation (requires Shapely)
    if HAS_SHAPELY_VIEWS and boundary_shape is not None:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if not boundary_shape.contains(ShapelyPoint(cx, cy)):
            box_geom = shapely_box(x1, y1, x2, y2)
            intersection = box_geom.intersection(boundary_shape)
            w, h = x2 - x1, y2 - y1
            if not intersection.is_empty and intersection.area >= 10:
                icx, icy = intersection.centroid.x, intersection.centroid.y
            else:
                # Box entirely outside polygon – fall back to polygon centroid
                icx, icy = boundary_shape.centroid.x, boundary_shape.centroid.y
            x1 = icx - w / 2.0
            x2 = icx + w / 2.0
            y1 = icy - h / 2.0
            y2 = icy + h / 2.0
            # Re-clamp after translation
            x1 = max(xmin + margin, min(x1, xmax - margin))
            x2 = max(xmin + margin, min(x2, xmax - margin))
            y1 = max(ymin + margin, min(y1, ymax - margin))
            y2 = max(ymin + margin, min(y2, ymax - margin))
            if x2 <= x1: x2 = x1 + 10
            if y2 <= y1: y2 = y1 + 10

    return x1, y1, x2, y2


def _python_fallback_align(boundary, boxes, types, edges, threshold):
    """
    Python fallback for MATLAB align_fp when MATLAB is not available.
    Clips boxes to fit within boundary and returns aligned boxes.
    """
    boxes = np.array(boxes)
    types = np.array(types)
    boundary = np.array(boundary)

    # Get boundary extents
    x_min = np.min(boundary[:, 0])
    x_max = np.max(boundary[:, 0])
    y_min = np.min(boundary[:, 1])
    y_max = np.max(boundary[:, 1])

    # Clip boxes to be within boundary with small margin (except balconies which should extend outside)
    margin = 5  # pixels margin from boundary
    clipped_boxes = []
    for i, box in enumerate(boxes):
        if len(box) >= 4:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            # Balconies (type 9) should extend outside - don't clip them
            if i < len(types) and int(types[i]) == 9:
                # Balconies can extend outside, but coordinates must be ordered correctly
                if x2 <= x1:
                    x2 = x1 + 10
                if y2 <= y1:
                    y2 = y1 + 10
                clipped_boxes.append([x1, y1, x2, y2])
            else:
                # Clip to boundary limits
                x1 = max(x_min + margin, min(x1, x_max - margin))
                x2 = max(x_min + margin, min(x2, x_max - margin))
                y1 = max(y_min + margin, min(y1, y_max - margin))
                y2 = max(y_min + margin, min(y2, y_max - margin))

                # Ensure x2 > x1 and y2 > y1
                if x2 <= x1:
                    x2 = x1 + 10
                if y2 <= y1:
                    y2 = y1 + 10

                clipped_boxes.append([x1, y1, x2, y2])
        else:
            clipped_boxes.append(box)

    boxes = np.array(clipped_boxes)

    # Simple ordering: sort by y-coordinate (top to bottom), then x-coordinate (left to right)
    if len(boxes) > 0:
        centers_x = np.array([(boxes[i][0] + boxes[i][2]) / 2 for i in range(len(boxes))])
        centers_y = np.array([(boxes[i][1] + boxes[i][3]) / 2 for i in range(len(boxes))])
        order = np.argsort(centers_y * 1000 + centers_x) + 1  # MATLAB uses 1-indexing
        order = order.reshape(-1, 1)
    else:
        order = np.array([[]], dtype=int)

    # Generate simple room boundaries (just bounding boxes)
    room_boundaries = []
    for box in boxes:
        if len(box) >= 4:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            # Create a rectangle as the room boundary
            boundary_poly = [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
                [x1, y1]
            ]
            room_boundaries.append(boundary_poly)
        else:
            room_boundaries.append([[]])

    return boxes.tolist(), order.tolist(), room_boundaries


def home(request):
    return render(request, "home.html", )


def Init(request):
    start = time.perf_counter()
    getTestData()
    getTrainData()
    loadMatlabEng()
    loadModel()
    loadRetrieval()
    end = time.perf_counter()
    print('Init(model+test+train+engine+retrieval) time: %s Seconds' % (end - start))

    return HttpResponse(None)


def loadMatlabEng():
    startengview = time.perf_counter()
    global engview
    if HAS_MATLAB:
        try:
            engview = matlab.engine.start_matlab()
            engview.addpath(r'./align_fp/', nargout=0)
            endengview = time.perf_counter()
            print(' matlab.engineview time: %s Seconds' % (endengview - startengview))
            print('MATLAB engine initialized successfully')
        except Exception as e:
            engview = None
            warnings.warn(f"Failed to initialize MATLAB engine: {e}. "
                         "Using Python fallback for alignment.", UserWarning)
    else:
        engview = None
        print('MATLAB not available - using Python fallback for alignment')


def loadRetrieval():
    global tf_train, centroids, clusters
    t1 = time.perf_counter()
    tf_train = np.load(r'C:\Users\hmbashir\source\Graph2plan\Interface\retrieval\tf_train.npy')
    centroids = np.load(r'C:\Users\hmbashir\source\Graph2plan\Interface\retrieval\centroids_train.npy')
    clusters = np.load(r'C:\Users\hmbashir\source\Graph2plan\Interface\retrieval\clusters_train.npy')
    t2 = time.perf_counter()
    print('load tf/centroids/clusters', t2 - t1)


def getTestData():
    start = time.perf_counter()
    global test_data, testNameList, trainNameList
 
    test_data = pickle.load(open(r'C:\Users\hmbashir\source\Graph2plan\Interface\static\Data\data_test_converted.pkl', 'rb'))
    # Strip trailing spaces from name lists to match image filenames
    test_data, testNameList, trainNameList = (
        test_data['data'],
        [str(n).strip() for n in test_data['testNameList']],
        [str(n).strip() for n in test_data['trainNameList']]
    )

    print(f"📊 Loaded {len(test_data)} test floor plans")
    print(f"📊 testNameList has {len(testNameList)} names: {testNameList[:10]}")
    print(f"📊 trainNameList has {len(trainNameList)} names: {trainNameList[:10]}")

    end = time.perf_counter()
    print('getTestData time: %s Seconds' % (end - start))


def getTrainData():
    start = time.perf_counter()
    global train_data, trainNameList, trainTF, train_data_eNum, train_data_rNum
    
    train_data = pickle.load(open(r'C:\Users\hmbashir\source\Graph2plan\Interface\static\Data\data_train_converted.pkl', 'rb'))
    train_data, trainNameList, trainTF = train_data['data'], list(train_data['nameList']), list(train_data['trainTF'])
    
    train_data_eNum = pickle.load(open(r'C:\Users\hmbashir\source\Graph2plan\Interface\static\Data\data_train_eNum.pkl', 'rb'))
    train_data_eNum = train_data_eNum['eNum']
    train_data_rNum = np.load(r'C:\Users\hmbashir\source\Graph2plan\Interface\static\Data\rNum_train.npy')

    end = time.perf_counter()
    print('getTrainData time: %s Seconds' % (end - start))


def loadModel():
    global model, train_data, trainNameList
    start = time.perf_counter()
    model = mltest.load_model()
    end = time.perf_counter()
    print('loadModel time: %s Seconds' % (end - start))
    start = time.perf_counter()
    # Use first available training sample for warmup, or skip if none available
    if len(trainNameList) > 0:
        test = train_data[0]
        mltest.test(model, FloorPlan(test, train=True))
        end = time.perf_counter()
        print('test Model time: %s Seconds' % (end - start))
    else:
        print('Skipping model warmup - no training data available')


def LoadTestBoundary(request):
    start = time.perf_counter()
    testName = request.GET.get('testName').split(".")[0]
    print(f"🔍 LoadTestBoundary called with testName={testName}")

    # Handle case where testName doesn't exist in testNameList (e.g., old RPLAN names)
    if testName not in testNameList:
        print(f"Warning: Test name '{testName}' not found in testNameList. Using first test floor plan: {testNameList[0]}")
        testName = testNameList[0]

    test_index = testNameList.index(testName)
    data = test_data[test_index]
    # Handle both mat_struct (dict-like) and object attribute access
    data_name = data['name'] if isinstance(data, dict) or hasattr(data, '__getitem__') else (data.name if hasattr(data, 'name') else 'unknown')
    print(f"   → Loading test_data[{test_index}], name={data_name}, boundary shape={data.boundary.shape}, rBoundary count={len(data.rBoundary) if hasattr(data, 'rBoundary') else 'N/A'}")
    data_js = {}
    data_js["door"] = str(data.boundary[0][0]) + "," + str(data.boundary[0][1]) + "," + str(
        data.boundary[1][0]) + "," + str(data.boundary[1][1])
    ex = ""
    for i in range(len(data.boundary)):
        ex = ex + str(data.boundary[i][0]) + "," + str(data.boundary[i][1]) + " "
    data_js['exterior'] = ex
    end = time.perf_counter()
    print('LoadTestBoundary time: %s Seconds' % (end - start))
    return HttpResponse(json.dumps(data_js), content_type="application/json")


def get_filter_func(mask, acc, num):
    filters = [
        None if not mask else (
            np.equal if acc[i] else np.greater_equal
        )
        for i in range(len(mask))
    ]

    def filter_func(data):
        for i in range(len(filters)):
            if (filters[i] is not None) and (not filters[i](data[i], num[i])): return False
        return True

    return filter_func


def filter_graph(graph_):
    filters = graph_

    def filter_graphfunc(data):
        sub = data - filters
        return ((sub >= 0).all())

    return filter_graphfunc


def compute_similarity_scores(candidate_rNum, candidate_eNum, requested_rNum, requested_edge,
                               tf_distances, mask=None, alpha=0.35, beta=0.35, gamma=0.30):
    """
    Compute similarity scores combining room count, edge structure, and boundary similarity.
    Lower scores indicate better matches.

    Args:
        candidate_rNum: Array of room counts for candidates (N x num_room_types)
        candidate_eNum: Array of edge structures for candidates (N x edge_dim) or None
        requested_rNum: Requested room counts (num_room_types,)
        requested_edge: Requested edge structure or None
        tf_distances: Turn function distances for boundary similarity (N,)
        mask: Boolean mask indicating which room types to consider (num_room_types,)
        alpha: Weight for room count similarity (default 0.35)
        beta: Weight for edge structure similarity (default 0.35, ignored if no edge data)
        gamma: Weight for boundary similarity (default 0.30)

    Returns:
        scores: Array of similarity scores (N,) - lower is better
    """
    # Normalize weights if edge data is missing
    if candidate_eNum is None or requested_edge is None:
        alpha_norm = alpha / (alpha + gamma)
        gamma_norm = gamma / (alpha + gamma)
        beta_norm = 0.0
    else:
        alpha_norm = alpha
        beta_norm = beta
        gamma_norm = gamma

    # Apply mask if provided
    if mask is not None:
        mask_array = np.array(mask).astype(bool)
        candidate_counts = candidate_rNum[:, mask_array]
        requested_counts = np.array(requested_rNum)[mask_array]
    else:
        candidate_counts = candidate_rNum
        requested_counts = np.array(requested_rNum)

    # Room count similarity (L1 distance)
    room_distances = np.sum(np.abs(candidate_counts - requested_counts), axis=1)

    # Edge structure similarity (L1 distance) if available
    if candidate_eNum is not None and requested_edge is not None:
        edge_distances = np.sum(np.abs(candidate_eNum - requested_edge), axis=1)
    else:
        edge_distances = np.zeros_like(room_distances)

    # Normalize distances to [0, 1] range
    max_room_dist = np.max(room_distances) if np.max(room_distances) > 0 else 1.0
    max_edge_dist = np.max(edge_distances) if np.max(edge_distances) > 0 else 1.0
    max_tf_dist = np.max(tf_distances) if np.max(tf_distances) > 0 else 1.0

    room_distances_norm = room_distances / max_room_dist
    edge_distances_norm = edge_distances / max_edge_dist
    tf_distances_norm = tf_distances / max_tf_dist

    # Weighted combination
    scores = (alpha_norm * room_distances_norm +
              beta_norm * edge_distances_norm +
              gamma_norm * tf_distances_norm)

    return scores


def calculate_room_match_percentage(candidate_rNum, requested_rNum, mask=None, exact_match=None):
    """
    Calculate percentage match based on how many requested rooms are present.

    Args:
        candidate_rNum: Room counts for a candidate (num_room_types,)
        requested_rNum: Requested room counts (num_room_types,)
        mask: Boolean mask indicating which room types to consider
        exact_match: Boolean array indicating which room types require exact match (not just >=)

    Returns:
        percentage: Match percentage (0-100)
    """
    candidate_counts = np.array(candidate_rNum)
    requested_counts = np.array(requested_rNum)

    # Apply mask to filter which room types to consider
    if mask is not None:
        mask_array = np.array(mask).astype(bool)
        # Only consider room types where mask is True AND requested count > 0
        active_mask = mask_array & (requested_counts > 0)

        if not np.any(active_mask):
            # No room requirements specified
            print("WARNING: No room requirements specified, returning 0% match")
            return 0.0

        candidate_counts = candidate_counts[active_mask]
        requested_counts = requested_counts[active_mask]

        # Apply same mask to exact_match array
        if exact_match is not None:
            exact_match_array = np.array(exact_match).astype(bool)[active_mask]
        else:
            exact_match_array = None
    else:
        # No mask provided, filter by requested counts > 0
        active_mask = requested_counts > 0

        if not np.any(active_mask):
            # No room requirements specified
            print("WARNING: No room requirements specified, returning 0% match")
            return 0.0

        candidate_counts = candidate_counts[active_mask]
        requested_counts = requested_counts[active_mask]

        # Apply same filter to exact_match array
        if exact_match is not None:
            exact_match_array = np.array(exact_match).astype(bool)[active_mask]
        else:
            exact_match_array = None

    total_requested = np.sum(requested_counts)

    # Debug logging
    print(f"Candidate counts (filtered): {candidate_counts}")
    print(f"Requested counts (filtered): {requested_counts}")
    print(f"Exact match required (filtered): {exact_match_array}")
    print(f"Total requested: {total_requested}")

    # Calculate matched counts based on distance from exact match
    # Penalize both excess and missing rooms
    # Formula: matched = max(0, requested - |candidate - requested|)
    # Example: requested=1, candidate=2 → matched = max(0, 1 - 1) = 0
    # Example: requested=2, candidate=2 → matched = max(0, 2 - 0) = 2
    differences = np.abs(candidate_counts - requested_counts)
    matched_counts = np.maximum(0, requested_counts - differences)

    total_matched = np.sum(matched_counts)

    print(f"Matched counts: {matched_counts}, Total matched: {total_matched}")

    percentage = (total_matched / total_requested) * 100.0
    print(f"Match percentage: {percentage}%")

    return percentage


def NumSearch(request):
    start = time.perf_counter()
    data_new = json.loads(request.GET.get("userInfo"))
    getTestData()
    testName = data_new[0].split(".")[0]

    # Handle case where testName doesn't exist in testNameList (e.g., old RPLAN names)
    if testName not in testNameList:
        print(f"Warning: Test name '{testName}' not found in testNameList. Using first test floor plan: {testNameList[0]}")
        testName = testNameList[0]

    test_index = testNameList.index(testName)
    topkList = []
    topkList.clear()
    data = test_data[test_index]

   
    multi_clusters=False
    loadRetrieval()
    getTestData()
    getTrainData()
    test_data_topk = rt.retrieval(data, 1000,multi_clusters)
    
    if len(data_new) > 1:
        roomactarr = data_new[1]
        roomexaarr = data_new[2]
        roomnumarr = [int(x) for x in data_new[3]]
        
        test_num = train_data_rNum[test_data_topk]
        filter_func = get_filter_func(roomactarr, roomexaarr, roomnumarr)
        indices = np.where(list(map(filter_func, test_num)))
        indices = list(indices)

        topkList.clear()

        # FALLBACK: If hard filter returns no results, use similarity scoring
        if len(indices[0]) == 0:
            print("NumSearch: Hard filter returned no results. Using similarity-based fallback.")

            # Compute TF distances for all candidates
            x, y = rt.compute_tf(data.boundary)
            y_sampled = rt.sample_tf(x, y, 1000)
            tf_distances = np.linalg.norm(y_sampled - tf_train[test_data_topk], axis=1)

            # Compute similarity scores
            scores = compute_similarity_scores(
                candidate_rNum=test_num,
                candidate_eNum=None,  # No edge data in NumSearch
                requested_rNum=roomnumarr,
                requested_edge=None,
                tf_distances=tf_distances,
                mask=roomactarr,
                alpha=0.5,  # Room count weight
                beta=0.0,   # No edge data
                gamma=0.5   # Boundary similarity weight
            )

            # Sort by similarity score (lower is better)
            sorted_indices = np.argsort(scores)
            topk = min(20, len(sorted_indices))

            for i in range(topk):
                candidate_idx = test_data_topk[sorted_indices[i]]
                floor_plan_name = str(trainNameList[int(candidate_idx)]) + ".png"

                # Calculate match percentage
                match_percentage = calculate_room_match_percentage(
                    test_num[sorted_indices[i]],
                    roomnumarr,
                    roomactarr,
                    roomexaarr
                )

                # Return as object with name and match percentage
                topkList.append({
                    "name": floor_plan_name,
                    "match": round(match_percentage, 1),
                    "fallback": True  # Indicates this used fallback matching
                })
        else:
            # Original hard filter logic
            if len(indices[0]) < 20:
                topk = len(indices[0])
            else:
                topk = 20

            for i in range(topk):
                floor_plan_name = str(trainNameList[int(test_data_topk[indices[0][i]])]) + ".png"

                # Calculate match percentage (should be 100% for hard filter results)
                match_percentage = calculate_room_match_percentage(
                    test_num[indices[0][i]],
                    roomnumarr,
                    roomactarr,
                    roomexaarr
                )

                # Return as object with name and match percentage
                topkList.append({
                    "name": floor_plan_name,
                    "match": round(match_percentage, 1),
                    "fallback": False  # Hard filter match
                })

    end = time.perf_counter()
    print('NumberSearch time: %s Seconds' % (end - start))

    # Return both old format (for backward compatibility) and new format (with metadata)
    response_data = {
        "floorPlans": [item["name"] for item in topkList],  # Old format for UI compatibility
        "metadata": topkList  # New format with match scores
    }
    return HttpResponse(json.dumps(response_data), content_type="application/json")


def FindTraindata(trainname):
    start = time.perf_counter()
    train_index = trainNameList.index(trainname)
    data = train_data[train_index]
    data_js = {}
    data_js["hsname"] = trainname

    data_js["door"] = str(data.boundary[0][0]) + "," + str(data.boundary[0][1]) + "," + str(
        data.boundary[1][0]) + "," + str(data.boundary[1][1])
    print("testboundary", data_js["door"])
    ex = ""
    for i in range(len(data.boundary)):
        ex = ex + str(data.boundary[i][0]) + "," + str(data.boundary[i][1]) + " "
    data_js['exterior'] = ex

    data_js["hsedge"] = [[int(u), int(v)] for u, v in data.edge[:, [0, 1]]]

    external = np.asarray(data.boundary)
    xmin, xmax = np.min(external[:, 0]), np.max(external[:, 0])
    ymin, ymax = np.min(external[:, 1]), np.max(external[:, 1])

    # Clip boxes to boundary (except balconies which should extend outside)
    margin = 5
    hsbox = []
    for x1, y1, x2, y2, cate in data.box[:]:
        # Balconies (type 9) should extend outside - don't clip them
        if int(cate) == 9:
            # Balconies can extend outside, but coordinates must be ordered correctly
            x1_bal, y1_bal, x2_bal, y2_bal = float(x1), float(y1), float(x2), float(y2)
            if x2_bal <= x1_bal:
                x2_bal = x1_bal + 10
            if y2_bal <= y1_bal:
                y2_bal = y1_bal + 10
            hsbox.append([[x1_bal, y1_bal, x2_bal, y2_bal], [mdul.room_label[int(cate)][1]]])
        else:
            # Clip to boundary limits
            x1_clipped = max(xmin + margin, min(float(x1), xmax - margin))
            x2_clipped = max(xmin + margin, min(float(x2), xmax - margin))
            y1_clipped = max(ymin + margin, min(float(y1), ymax - margin))
            y2_clipped = max(ymin + margin, min(float(y2), ymax - margin))
            # Ensure x2 > x1 and y2 > y1
            if x2_clipped <= x1_clipped:
                x2_clipped = x1_clipped + 10
            if y2_clipped <= y1_clipped:
                y2_clipped = y1_clipped + 10
            hsbox.append([[x1_clipped, y1_clipped, x2_clipped, y2_clipped], [mdul.room_label[int(cate)][1]]])

    area_ = (ymax - ymin) * (xmax - xmin)
    
    data_js["rmsize"] = [
        [[20 * math.sqrt((float(x2) - float(x1)) * (float(y2) - float(y1)) / float(area_))], [mdul.room_label[int(cate)][1]]]
        for
        x1, y1, x2, y2, cate in data.box[:]]
   

    box_order = data.order
    
    # Reorder boxes by size (largest first, smallest last)
    # This ensures smaller rooms are drawn on top when overlapping
    box_sizes = []
    for i in range(len(box_order)):
        box_idx = int(float(box_order[i])) - 1
        box = hsbox[box_idx]
        x1, y1, x2, y2 = box[0][0], box[0][1], box[0][2], box[0][3]
        area = (x2 - x1) * (y2 - y1)
        box_sizes.append((area, i, box))
    
    # Sort by area (descending - largest first)
    box_sizes.sort(key=lambda x: x[0], reverse=True)
    
    data_js["hsbox"] = []
    for area, original_idx, box in box_sizes:
        data_js["hsbox"].append(box)

    data_js["rmpos"] = [[int(cate), str(mdul.room_label[int(cate)][1]), float((x1 + x2) / 2), float((y1 + y2) / 2)] for
                        x1, y1, x2, y2, cate in data.box[:]]
    end = time.perf_counter()
    print('find train data time: %s Seconds' % (end - start))
    return data_js


def LoadTrainHouse(request):
    trainname = request.GET.get("roomID").split(".")[0]
    data_js = FindTraindata(trainname)
    return HttpResponse(json.dumps(data_js), content_type="application/json")


'''
 transfer the graph of the training data into the graph of the test data
'''


def TransGraph(request):
    start = time.perf_counter()
    userInfo = request.GET.get("userInfo")
    testname = userInfo.split(',')[0]
    trainname = request.GET.get("roomID")
    mlresult = mltest.get_userinfo(testname, trainname)

    fp_end = mlresult

    # Use absolute path for static directory
    static_dir = os.path.join(settings.BASE_DIR, 'static')
    os.makedirs(static_dir, exist_ok=True)  # Ensure directory exists
    mat_filename = userInfo.split(',')[0].split('.')[0] + ".mat"
    mat_filepath = os.path.join(static_dir, mat_filename)
    sio.savemat(mat_filepath, {"data": fp_end.data})

    data_js = {}
    # fp_end  hsedge
    data_js["hsedge"] = (fp_end.get_triples(tensor=False)[:, [0, 2, 1]]).astype(float).tolist()

    # fp_rmsize
    external = np.asarray(fp_end.data.boundary)
    xmin, xmax = np.min(external[:, 0]), np.max(external[:, 0])
    ymin, ymax = np.min(external[:, 1]), np.max(external[:, 1])
    area_ = (ymax - ymin) * (xmax - xmin)
    boundary_shape = ShapelyPolygon(external[:, :2]) if HAS_SHAPELY_VIEWS else None

    # Clip boxes to boundary (polygon-aware, same margin as AdjustGraph)
    margin = 5
    clipped_boxes = []
    for box in fp_end.data.box[:]:
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        cate = int(box[4]) if len(box) > 4 else 0
        if cate == 9:  # Balconies can extend outside boundary
            if x2 <= x1: x2 = x1 + 10
            if y2 <= y1: y2 = y1 + 10
        else:
            x1, y1, x2, y2 = _clip_box_to_polygon(x1, y1, x2, y2, xmin, xmax, ymin, ymax, margin, boundary_shape)
        clipped_boxes.append([x1, y1, x2, y2, cate])

    data_js["rmsize"] = [
        [[20 * math.sqrt((x2 - x1) * (y2 - y1) / float(area_))], [mdul.room_label[int(cate)][1]]]
        for x1, y1, x2, y2, cate in clipped_boxes]
    # fp_end rmpos

    rooms = fp_end.get_rooms(tensor=False)

    center = [[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2, _ in clipped_boxes]

    # boxes_pred
    data_js["rmpos"] = []
    for k in range(len(center)):
        node = float(rooms[k]), mdul.room_label[int(rooms[k])][1], center[k][0], center[k][1], float(k)
        data_js["rmpos"].append(node)

    test_index = testNameList.index(testname.split(".")[0])
    data = test_data[test_index]
    ex = ""
    for i in range(len(data.boundary)):
        ex = ex + str(data.boundary[i][0]) + "," + str(data.boundary[i][1]) + " "
    data_js['exterior'] = ex
    data_js["door"] = str(data.boundary[0][0]) + "," + str(data.boundary[0][1]) + "," + str(
        data.boundary[1][0]) + "," + str(data.boundary[1][1])
    end = time.perf_counter()
    print('TransGraph time: %s Seconds' % (end - start))
    return HttpResponse(json.dumps(data_js), content_type="application/json")


def AdjustGraph(request):
    start = time.perf_counter()
    print("\n" + "="*80)
    print("🚀 [VIEWS] AdjustGraph API called")
    print("="*80)
    
    # newNode index-typename-cx-cy
    # oldNode index-typename-cx-cy
    # newEdge u-v
    NewGraph = json.loads(request.GET.get("NewGraph"))
    testname = request.GET.get("userRoomID")
    trainname = request.GET.get("adptRoomID")
    
    print(f"📥 [VIEWS] Request Parameters:")
    print(f"   → testname (user boundary): {testname}")
    print(f"   → trainname (template): {trainname}")
    print(f"   → NewGraph structure: {len(NewGraph)} elements")
    if len(NewGraph) > 0:
        print(f"   → NewGraph[0] (nodes): {NewGraph[0][:3] if len(NewGraph[0]) > 3 else NewGraph[0]}... ({len(NewGraph[0])} total nodes)")
    if len(NewGraph) > 1:
        print(f"   → NewGraph[1] (edges): {NewGraph[1][:3] if len(NewGraph[1]) > 3 else NewGraph[1]}... ({len(NewGraph[1])} total edges)")
    
    print(f"\n🧠 [VIEWS] Calling model inference...")
    s = time.perf_counter()
    try:
        mlresult = mltest.get_userinfo_adjust(testname, trainname, NewGraph)
        e = time.perf_counter()
        print(f'✅ [VIEWS] Model inference completed: {e - s:.3f} seconds')
        print(f"   → mlresult type: {type(mlresult)}, length: {len(mlresult) if isinstance(mlresult, (list, tuple)) else 'N/A'}")
    except Exception as ex:
        print(f"❌ [VIEWS] Model inference FAILED: {type(ex).__name__}: {ex}")
        import traceback
        traceback.print_exc()
        raise
    fp_end = mlresult[0]
    global boxes_pred
    boxes_pred = mlresult[1]
    
    print(f"\n📦 [VIEWS] Processing model output:")
    print(f"   → fp_end type: {type(fp_end)}")
    print(f"   → boxes_pred shape: {boxes_pred.shape if hasattr(boxes_pred, 'shape') else len(boxes_pred)}")
    print(f"   → boxes_pred dtype: {boxes_pred.dtype if hasattr(boxes_pred, 'dtype') else type(boxes_pred)}")
    print(f"   → boxes_pred sample: {boxes_pred[:2] if len(boxes_pred) > 0 else 'empty'}")
    
    data_js = {}
    triples = fp_end.get_triples(tensor=False)[:, [0, 2, 1]]
    data_js["hsedge"] = triples.astype(float).tolist()
    print(f"   → hsedge (triples) count: {len(data_js['hsedge'])}")
  
    rooms = fp_end.get_rooms(tensor=False)
    center = [[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in fp_end.data.box[:, :4]]

    box_order = mlresult[2]
    '''
    handle the information of the room boxes 
    boxes_pred: the prediction from net
    box_order: The order in which boxes are drawn

    '''
    room = []
    for o in range(len(box_order)):
        room.append(float((rooms[int(float(box_order[o][0])) - 1])))
    boxes_end = []
    for i in range(len(box_order)):
        tmp = []
        for j in range(4):
            tmp.append(float(boxes_pred[int(float(box_order[i][0])) - 1][j]))
        boxes_end.append(tmp)

    # Get boundary for clipping
    print(f"🔍 AdjustGraph: Loading boundary for testname={testname}")
    test_index = testNameList.index(testname.split(".")[0])
    test_data_item = test_data[test_index]
    # Handle both mat_struct (dict-like) and object attribute access
    data_name = test_data_item['name'] if isinstance(test_data_item, dict) or hasattr(test_data_item, '__getitem__') else (test_data_item.name if hasattr(test_data_item, 'name') else 'unknown')
    print(f"   → test_data[{test_index}], name={data_name}, boundary shape={test_data_item.boundary.shape}, rBoundary count={len(test_data_item.rBoundary) if hasattr(test_data_item, 'rBoundary') else 'N/A'}")

    external = np.asarray(test_data_item.boundary)
    xmin, xmax = np.min(external[:, 0]), np.max(external[:, 0])
    ymin, ymax = np.min(external[:, 1]), np.max(external[:, 1])
    boundary_shape = ShapelyPolygon(external[:, :2]) if HAS_SHAPELY_VIEWS else None

    # Clip boxes to boundary (except balconies which should extend outside)
    margin = 5
    clipped_boxes_end = []
    for i, box in enumerate(boxes_end):
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        # Balconies (type 9) should extend outside - don't clip them
        if int(room[i]) == 9:
            # Balconies can extend outside, but coordinates must be ordered correctly
            if x2 <= x1:
                x2 = x1 + 10
            if y2 <= y1:
                y2 = y1 + 10
            clipped_boxes_end.append([x1, y1, x2, y2])
        else:
            x1, y1, x2, y2 = _clip_box_to_polygon(x1, y1, x2, y2, xmin, xmax, ymin, ymax, margin, boundary_shape)
            clipped_boxes_end.append([x1, y1, x2, y2])

    boxes_end = clipped_boxes_end
    print(f"   → Clipped {len(boxes_end)} boxes to boundary (polygon-aware)")

    # Sort rooms by area (largest first) to ensure proper layering
    # Larger rooms rendered first (behind), smaller rooms last (on top)
    room_sizes = []
    for k in range(len(boxes_end)):
        x1, y1, x2, y2 = boxes_end[k][0], boxes_end[k][1], boxes_end[k][2], boxes_end[k][3]
        area = (x2 - x1) * (y2 - y1)
        # Store (area, original_index, box, room_type, box_order_entry)
        room_sizes.append((area, k, boxes_end[k], room[k], box_order[k]))

    # Sort by area (descending - largest first)
    room_sizes.sort(key=lambda x: x[0], reverse=True)

    # Rebuild arrays in sorted order (maintaining parallel structure)
    boxes_end = [item[2] for item in room_sizes]
    room = [item[3] for item in room_sizes]
    box_order = [item[4] for item in room_sizes]
    print(f"   → Sorted {len(boxes_end)} rooms by area for proper layering")

    data_js['roomret'] = []
    for k in range(len(room)):
        room_data = boxes_end[k], [mdul.room_label[int(room[k])][1]], box_order[k][0] - 1
        data_js['roomret'].append(room_data)
    print(f"   → Generated {len(data_js['roomret'])} room entries for response")

    # change the box size
    global relbox
    relbox = data_js['roomret']
    global reledge
    reledge = data_js["hsedge"]

    # Use test_boundary from earlier (line 708: data = test_data[test_index])
    ex = ""
    for i in range(len(external)):
        ex = ex + str(external[i][0]) + "," + str(external[i][1]) + " "
    data_js['exterior'] = ex
    data_js["door"] = str(external[0][0]) + "," + str(external[0][1]) + "," + str(
        external[1][0]) + "," + str(external[1][1])
    area_ = (ymax - ymin) * (xmax - xmin)
    data_js['rmsize'] = []
    for i in range(len(data_js['roomret'])):
        size_value = 20 * math.sqrt((float(data_js['roomret'][i][0][2]) - float(data_js['roomret'][i][0][0])) * (
                float(data_js['roomret'][i][0][3]) - float(data_js['roomret'][i][0][1])) / float(area_))
        rmsize = [[size_value], [data_js["roomret"][i][1][0]]]
        data_js["rmsize"].append(rmsize)

    data_js["rmpos"] = []

    newGraph = NewGraph[0]
    for i in range(len(data_js['roomret'])):
        for k in range(len(newGraph)):
            if (data_js['roomret'][i][1][0] == newGraph[k][1]):
                x_center = int((data_js['roomret'][i][0][0] + data_js['roomret'][i][0][2]) / 2)
                y_center = int((data_js['roomret'][i][0][1] + data_js['roomret'][i][0][3]) / 2)
                x_graph = newGraph[k][2]
                y_graph = newGraph[k][3]
                room_idx = int(data_js['roomret'][i][2])
                if ((int(x_graph - 30) < x_center < int(x_graph + 30))):
                    node = float(rooms[room_idx]), newGraph[k][1], x_center, y_center, float(
                        newGraph[k][0])
                    data_js["rmpos"].append(node)
                    newGraph.pop(k)
                    break
                if ((int(y_graph - 30) < y_center < int(y_graph + 30))):
                    node = float(rooms[room_idx]), newGraph[k][1], x_center, y_center, float(
                        newGraph[k][0])
                    data_js["rmpos"].append(node)
                    newGraph.pop(k)

                    break
    
    fp_end.data = add_dw_fp(fp_end.data)

    # Populate indoor with room boundary polygons (rBoundary)
    # This makes the Layout view match the thumbnail images
    data_js["indoor"] = []
    if hasattr(fp_end.data, 'rBoundary') and fp_end.data.rBoundary:
        # Use rBoundary from generated floor plan if available
        for rb in fp_end.data.rBoundary:
            if isinstance(rb, np.ndarray) and len(rb) > 0:
                coords_str = " ".join([f"{x},{y}" for x, y in rb])
                data_js["indoor"].append(coords_str)
    elif hasattr(test_data_item, 'rBoundary') and test_data_item.rBoundary:
        # Fallback to test data rBoundary if generation doesn't have it
        for rb in test_data_item.rBoundary:
            if isinstance(rb, (np.ndarray, list)) and len(rb) > 0:
                rb_array = np.array(rb) if not isinstance(rb, np.ndarray) else rb
                coords_str = " ".join([f"{x},{y}" for x, y in rb_array])
                data_js["indoor"].append(coords_str)

    boundary = test_data_item.boundary
    
    isNew = boundary[:, 3]
    frontDoor = boundary[[0, 1]]  
    frontDoor = frontDoor[:, [0, 1]]  
    frontsum = frontDoor.sum(axis=1).tolist()
    idx = frontsum.index(min(frontsum))
    wallThickness = 3
    if idx == 1:
        frontDoor = frontDoor[[1, 0], :]
    orient = boundary[0][2]
    if orient == 0 or orient == 2:
        frontDoor[0][0] = frontDoor[0][0] + wallThickness / 4
        frontDoor[1][0] = frontDoor[1][0] - wallThickness / 4
    if orient == 1 or orient == 3:
        frontDoor[0][1] = frontDoor[0][1] + wallThickness / 4
        frontDoor[1][1] = frontDoor[1][1] - wallThickness / 4
    

    data_js["windows"] = []
    for indx, x, y, w, h, r in fp_end.data.windows:
        if w != 0:
            tmp = [x + 2, y - 2, w - 2, 4]
            data_js["windows"].append(tmp)
        if h != 0:
            tmp = [x - 2, y, 4, h]
            data_js["windows"].append(tmp)
    data_js["windowsline"] = []
    for indx, x, y, w, h, r in fp_end.data.windows:
        if w != 0:
            tmp = [x + 2, y, w + x, y]
            data_js["windowsline"].append(tmp)
        if h != 0:
            tmp = [x, y, x, h + y]
            data_js["windowsline"].append(tmp)

    # Save as .mat file
    mat_filename = "./static/" + testname.split(',')[0].split('.')[0] + ".mat"
    sio.savemat(mat_filename, {"data": fp_end.data})
    print(f"\n💾 [VIEWS] Saved floor plan data to {mat_filename}")

    # Optional: Auto-save as DXF file (disabled by default)
    if ENABLE_AUTO_DXF_EXPORT and HAS_DXF_EXPORT:
        try:
            dxf_filename = "./static/" + testname.split(',')[0].split('.')[0] + ".dxf"
            success = save_floorplan_dxf(
                fp_end.data,
                dxf_filename,
                scale=DXF_SCALE,
                wall_thickness=DXF_WALL_THICKNESS,
                include_labels=True,
                include_dimensions=False
            )
            if success:
                print(f"💾 [VIEWS] ✓ Auto-saved DXF file: {dxf_filename}")
            else:
                print(f"💾 [VIEWS] ✗ Failed to auto-save DXF file")
        except Exception as e:
            print(f"💾 [VIEWS] Warning: DXF auto-export failed: {e}")

    end = time.perf_counter()
    print(f"\n✅ [VIEWS] AdjustGraph completed successfully: {end - start:.3f} seconds")
    print(f"📤 [VIEWS] Response JSON structure:")
    print(f"   → roomret: {len(data_js.get('roomret', []))} rooms")
    print(f"   → hsedge: {len(data_js.get('hsedge', []))} edges")
    print(f"   → indoor: {len(data_js.get('indoor', []))} room boundaries")
    print(f"   → windows: {len(data_js.get('windows', []))} windows")
    print(f"   → rmsize: {len(data_js.get('rmsize', []))} room sizes")
    print(f"   → rmpos: {len(data_js.get('rmpos', []))} room positions")
    print("="*80 + "\n")
    return HttpResponse(json.dumps(data_js), content_type="application/json")


def RelBox(request):
    id = request.GET.get("selectRect")
    print(id)
    global relbox
    global reledge
    rdirgroup=get_dir(id,relbox,reledge)
    return HttpResponse(json.dumps(rdirgroup), content_type="application/json")

def get_dir(id,relbox,reledge):
    rel = []
    selectindex = int(id.split("_")[1])
    select = np.zeros(4).astype(int)
    for i in range(len(relbox)):
        a = math.ceil(relbox[i][0][0]), math.ceil(relbox[i][0][1]), math.ceil(relbox[i][0][2]), math.ceil(
            relbox[i][0][3]), int(relbox[i][2])
        rel.append(a)
        if (selectindex == int(relbox[i][2])):
            # select:x1,x0,y0,y1.relbox:x0,y0,x1,y1
            select[0] = math.ceil(relbox[i][0][2])
            select[1] = math.ceil(relbox[i][0][0])
            select[2] = math.ceil(relbox[i][0][1])
            select[3] = math.ceil(relbox[i][0][3])
    rel = np.array(rel)
    df = pd.DataFrame({'x0': rel[:, 0], 'y0': rel[:, 1], 'x1': rel[:, 2], 'y1': rel[:, 3], 'rindex': rel[:, 4]})
    group_label = [(0, 'x1', "right"),
                   (1, 'x0', "left"),
                   (2, 'y0', "top"),
                   (3, 'y1', "down")]
    dfgroup = []
    for i in range(len(group_label)):
        dfgroup.append(df.groupby(group_label[i][1], as_index=True).get_group(name=select[i]))
    rdirgroup = []
    for i in range(len(dfgroup)):
        dir = dfgroup[i]
        rdir = []
        for k in range(len(dir)):
            idx = (dir.loc[dir['rindex'] == (dir.iloc[[k]].values)[0][4]].index.values)[0]
            rdir.append(relbox[idx][1][0].__str__() + "_" + (dir.iloc[[k]].values)[0][4].__str__())
        rdirgroup.append(rdir)
    reledge = np.array(reledge)
    data1 = reledge[np.where((reledge[:, [0]] == selectindex))[0]]
    data2 = reledge[np.where((reledge[:, [1]] == selectindex))[0]]
    reledge1 = np.vstack((data1, data2))
    return rdirgroup
def Save_Editbox(request):
    global indxlist,boxes_pred
    NewGraph = json.loads(request.GET.get("NewGraph"))
    NewLay = json.loads(request.GET.get("NewLay"))
    userRoomID = request.GET.get("userRoomID")
    adptRoomID = request.GET.get("adptRoomID")
    
    NewLay=np.array(NewLay)
    NewLay=NewLay[np.argsort(NewLay[:, 1])][:,2:]
    NewLay=NewLay.astype(float).tolist()

    test_index = testNameList.index(userRoomID.split(".")[0])
    test_ = test_data[test_index]
    
    Boundary = test_.boundary
    boundary=[[float(x),float(y),float(z),float(k)] for x,y,z,k in list(Boundary)]
    test_fp =FloorPlan(test_)

    train_index = trainNameList.index(adptRoomID.split(".")[0])
    train_ =train_data[train_index]
    train_fp =FloorPlan(train_,train=True)
    fp_end = test_fp.adapt_graph(train_fp)
    fp_end.adjust_graph()
    newNode = NewGraph[0]
    newEdge = NewGraph[1]
    oldNode = NewGraph[2]
    temp = []
    for newindx, newrmname, newx, newy,scalesize in newNode:
        for _, oldrmname, oldx, oldy, oldindx in oldNode:
            if (int(newindx) == oldindx):
                tmp=int(newindx), (newx - oldx), ( newy- oldy),float(scalesize)
                temp.append(tmp)
    newbox=[]
    if mltest.adjust==True and boxes_pred is not None:
        oldbox = []
        for i in range(len(boxes_pred)):
            indxtmp=[boxes_pred[i][0],boxes_pred[i][1],boxes_pred[i][2],boxes_pred[i][3],boxes_pred[i][0]]
            oldbox.append(indxtmp)
    else:
        indxlist=[]
        oldbox=fp_end.data.box.tolist()
        for i in range(len(oldbox)):
            indxlist.append([oldbox[i][4]])
        indxlist=np.array(indxlist)
        adjust=True
    oldbox=fp_end.data.box.tolist()
    X=0
    Y=0
    for i in range(len(oldbox)):
        X= X+(oldbox[i][2]-oldbox[i][0])
        Y= Y+(oldbox[i][3]-oldbox[i][1])
    x_ave=(X/len(oldbox))/2
    y_ave=(Y/len(oldbox))/2

    index_mapping = {}
    #  The room that already exists
    #  Move: Just by the distance
    for newindx, tempx, tempy,scalesize in temp:
        index_mapping[newindx] = len(newbox)
        tmpbox=[]
        scalesize = int(scalesize)
        if scalesize<1:
            scale = math.sqrt(scalesize)
            scalex = (oldbox[newindx][2] - oldbox[newindx][0]) * (1 - scale) / 2
            scaley = (oldbox[newindx][3] - oldbox[newindx][1]) * (1 - scale) / 2
            tmpbox = [(oldbox[newindx][0] + tempx) + scalex, (oldbox[newindx][1] + tempy)+scaley,
                      (oldbox[newindx][2] + tempx) - scalex, (oldbox[newindx][3] + tempy) - scaley, oldbox[newindx][4]]
        if scalesize == 1:
            tmpbox = [(oldbox[newindx][0] + tempx) , (oldbox[newindx][1] + tempy) ,(oldbox[newindx][2] + tempx), (oldbox[newindx][3] + tempy), oldbox[newindx][4]]

        if scalesize>1:
            scale=math.sqrt(scalesize)
            scalex = (oldbox[newindx][2] - oldbox[newindx][0]) * ( scale-1) / 2
            scaley = (oldbox[newindx][3] - oldbox[newindx][1]) * (scale-1) / 2
            tmpbox = [(oldbox[newindx][0] + tempx) - scalex, (oldbox[newindx][1] + tempy) - scaley,
                      (oldbox[newindx][2] + tempx) + scalex, (oldbox[newindx][3] + tempy) + scaley, oldbox[newindx][4]]

        newbox.append(tmpbox)

    #  The room just added
    #  Move: The room node with the average size of the existing room
    for newindx, newrmname, newx, newy,scalesize in newNode:
        if int(newindx)>(len(oldbox)-1):
            scalesize=int(scalesize)
            index_mapping[int(newindx)] = (len(newbox))
            tmpbox=[]
            if scalesize < 1:
                scale = math.sqrt(scalesize)
                scalex = x_ave * (1 - scale) / 2
                scaley = y_ave* (1 - scale) / 2
                tmpbox = [(newx-x_ave) +scalex,(newy-y_ave) +scaley,(newx+x_ave)-scalex,(newy+y_ave)-scaley,vocab['object_name_to_idx'][newrmname]]

            if scalesize == 1:
                tmpbox = [(newx - x_ave), (newy - y_ave), (newx + x_ave), (newy + y_ave),vocab['object_name_to_idx'][newrmname]]
            if scalesize > 1:
                scale = math.sqrt(scalesize)
                scalex = x_ave * (scale - 1) / 2
                scaley = y_ave * (scale - 1) / 2
                tmpbox = [(newx-x_ave) - scalex, (newy-y_ave)  - scaley,(newx+x_ave) + scalex, (newy+y_ave) + scaley,vocab['object_name_to_idx'][newrmname]]
            # tmpboxin = [(newx-x_ave) ,(newy-y_ave) ,(newx+x_ave) ,(newy+y_ave) ,vocab['object_name_to_idx'][newrmname]]
            # print(tmpboxin)
            # print(tmpbox)
            # print(scalesize)
            newbox.append(tmpbox)

    fp_end.data.box=np.array(newbox)
    
    adjust_Edge=[]
    for u, v in newEdge:
        tmp=[index_mapping[int(u)],index_mapping[int(v)], 0]
        adjust_Edge.append(tmp)
    fp_end.data.edge=np.array(adjust_Edge)
    rType = fp_end.get_rooms(tensor=False)

    rEdge = fp_end.get_triples(tensor=False)[:, [0, 2, 1]]
    Edge = [[float(u), float(v), float(type2)] for u, v, type2 in rEdge]
    Box=NewLay
    fp_end.data.boundary =np.array(boundary)
    fp_end.data.rType =np.array(rType).astype(int)
    fp_end.data.refineBox=np.array(Box)
    fp_end.data.rEdge=np.array(Edge)

    # Try Python refinement first (HAS_PYTHON_REFINEMENT is set at module init)
    refinement_method = "none"

    if HAS_PYTHON_REFINEMENT:
        try:
            print(f"[Refinement] Using Python geometric refinement for {userRoomID}")
            box_out, box_order, rBoundary = align_fp_python(
                boundary=np.array(boundary),
                boxes=np.array(Box),
                room_types=rType,
                edges=np.array(Edge),
                fp_id=userRoomID,  # FIX: Use userRoomID instead of undefined fp_id
                threshold=8.0,
                draw_result=False
            )
            refinement_method = "python"
            print(f"[Refinement] ✓ Python refinement successful! Got {len(box_out)} boxes")

        except Exception as e:
            print(f"[Refinement] ✗ Python refinement failed: {e}")
            print("[Refinement] Falling back to MATLAB...")

    if refinement_method == "none" and engview is not None and HAS_MATLAB:
        try:
            print(f"[Refinement] Using MATLAB alignment for {userRoomID}")
            # Use MATLAB alignment
            boundary_mat = matlab.double(boundary)
            rType_mat = matlab.double(rType.tolist())
            Edge_mat = matlab.double(Edge)
            Box_mat = matlab.double(Box)
            box_refine = engview.align_fp(boundary_mat, Box_mat,  rType_mat, Edge_mat, 18, False, nargout=3)
            box_out = box_refine[0]
            box_order = box_refine[1]
            rBoundary = box_refine[2]
            refinement_method = "matlab"
            print(f"[Refinement] ✓ MATLAB refinement successful!")

        except Exception as e:
            print(f"[Refinement] ✗ MATLAB failed: {e}")

    if refinement_method == "none":
        print(f"[Refinement] Using basic Python fallback for {userRoomID}")
        # Use Python fallback as last resort
        box_out, box_order, rBoundary = _python_fallback_align(boundary, Box, rType.tolist(), Edge, 18)
        refinement_method = "fallback"

    fp_end.data.newBox = np.array(box_out)
    fp_end.data.order = np.array(box_order)
    fp_end.data.rBoundary = [np.array(rb) for rb in rBoundary]
    fp_end.data = add_dw_fp(fp_end.data)

    # Save as .mat file
    mat_filepath = "./static/" + userRoomID + ".mat"
    sio.savemat(mat_filepath, {"data": fp_end.data})
    print(f"[Save] Saved .mat file: {mat_filepath}")

    # Optional: Auto-save as DXF file (disabled by default, controlled by separate button)
    if ENABLE_AUTO_DXF_EXPORT and HAS_DXF_EXPORT:
        try:
            dxf_filepath = "./static/" + userRoomID + ".dxf"
            success = save_floorplan_dxf(
                fp_end.data,
                dxf_filepath,
                scale=DXF_SCALE,
                wall_thickness=DXF_WALL_THICKNESS,
                include_labels=True,
                include_dimensions=False
            )
            if success:
                print(f"[Save] ✓ Auto-saved DXF file: {dxf_filepath}")
            else:
                print(f"[Save] ✗ Failed to auto-save DXF file")
        except Exception as e:
            print(f"[Save] Warning: DXF auto-export failed: {e}")

    flag=1
    return HttpResponse(json.dumps(flag), content_type="application/json")


def Refine_Floorplan(request):
    """
    NEW ENDPOINT: Manual refinement trigger for debugging.

    This endpoint can be called from a button in the interface to
    re-run refinement on an existing floor plan.

    GET parameters:
        - userRoomID: Floor plan identifier
        - threshold: (optional) Refinement threshold in pixels (default: 8.0)
        - method: (optional) Force method: 'python', 'matlab', or 'fallback'
        - expand_living: (optional) If 'true', expand living room to fill boundary (default: false)

    Returns:
        JSON with refinement results and statistics
    """
    userRoomID = request.GET.get("userRoomID")
    threshold = float(request.GET.get("threshold", "8.0"))
    force_method = request.GET.get("method", None)  # Optional: force a specific method
    expand_living = request.GET.get("expand_living", "false").lower() == "true"  # Optional: expand living room

    if not userRoomID:
        return JsonResponse({
            "success": False,
            "error": "userRoomID parameter required"
        }, status=400)

    # Capture all print() output so it can be downloaded from the interface.
    _tee = _TeeBuffer(sys.stdout)
    _old_stdout = sys.stdout
    sys.stdout = _tee

    try:
        # Load the saved floor plan data
        # userRoomID should be the base name (e.g., "14926"), so we need to add .mat extension
        mat_path = f"./static/{userRoomID}.mat"
        if not os.path.exists(mat_path):
            return JsonResponse({
                "success": False,
                "error": f"Floor plan {userRoomID} not found at {mat_path}"
            }, status=404)

        data = sio.loadmat(mat_path)
        fp_data = data['data'][0, 0]

        # Extract necessary data
        boundary = fp_data['boundary']
        room_types = fp_data['rType'].flatten()
        edges = fp_data['rEdge']
        
        # ALWAYS keep original boxes for reference (used in Pass 2 pre-scan)
        original_boxes = fp_data['refineBox']

        # NEW: Track refinement pass (1, 2, or 3)
        # Store at top level of data dict to avoid structured array issues
        if 'refinement_pass' in data:
            current_pass = int(data['refinement_pass'][0, 0]) if data['refinement_pass'].size > 0 else 1
        else:
            current_pass = 1

        # Clamp to valid range (1–7); anything out of range restarts at 1
        if current_pass > 7 or current_pass < 1:
            current_pass = 1

        # CRITICAL: On Pass 2 or 3, use the refined boxes from previous passes, not the original boxes!
        if current_pass >= 2 and 'newBox' in fp_data.dtype.names and fp_data['newBox'].size > 0:
            boxes = fp_data['newBox']  # Use refined boxes from previous pass
            print(f"  Loading refined boxes from Pass {current_pass-1} for Pass {current_pass} refinement")
        else:
            boxes = original_boxes  # Original boxes (for Pass 1 or if no refined boxes exist)

        print(f"\n[Manual Refine] Processing {userRoomID}")
        print(f"  Boxes: {len(boxes)}, Threshold: {threshold}px, Force method: {force_method}")
        print(f"  Refinement Pass: {current_pass}/7")
        print(f"  Expand Living Room: {expand_living}")

        # Run refinement based on method preference
        refinement_method = "none"
        box_out = None
        box_order = None
        rBoundary = None
        error_msg = None

        # Try Python refinement
        if (force_method is None or force_method == "python") and HAS_PYTHON_REFINEMENT:
            try:
                print(f"[Manual Refine] Trying Python refinement (pass {current_pass})...")
                box_out, box_order, rBoundary = align_fp_python(
                    boundary=boundary,
                    boxes=boxes,
                    room_types=room_types,
                    edges=edges,
                    fp_id=f"{userRoomID}_manual",
                    threshold=threshold,
                    draw_result=False,
                    refinement_pass=current_pass,  # Pass the current pass number
                    expand_living_room=expand_living,  # Pass living room expansion flag
                    original_boxes=original_boxes  # Pass original boxes for pre-scan
                )
                refinement_method = "python"
                print(f"[Manual Refine] ✓ Python refinement successful!")

            except Exception as e:
                error_msg = str(e)
                print(f"[Manual Refine] ✗ Python refinement failed: {e}")

        # Try MATLAB if Python failed or was forced
        if refinement_method == "none" and (force_method is None or force_method == "matlab") and HAS_MATLAB:
            try:
                print(f"[Manual Refine] Trying MATLAB refinement...")
                boundary_mat = matlab.double(boundary.tolist())
                rType_mat = matlab.double(room_types.tolist())
                Edge_mat = matlab.double(edges.tolist())
                Box_mat = matlab.double(boxes.tolist())
                box_refine = engview.align_fp(boundary_mat, Box_mat, rType_mat, Edge_mat, int(threshold), False, nargout=3)
                box_out = box_refine[0]
                box_order = box_refine[1]
                rBoundary = box_refine[2]
                refinement_method = "matlab"
                print(f"[Manual Refine] ✓ MATLAB refinement successful!")

            except Exception as e:
                error_msg = str(e)
                print(f"[Manual Refine] ✗ MATLAB refinement failed: {e}")

        # Try fallback if everything else failed
        if refinement_method == "none":
            try:
                print(f"[Manual Refine] Using basic Python fallback...")
                box_out, box_order, rBoundary = _python_fallback_align(
                    boundary, boxes, room_types.tolist(), edges, threshold
                )
                refinement_method = "fallback"
                print(f"[Manual Refine] ✓ Fallback refinement complete")

            except Exception as e:
                error_msg = str(e)
                print(f"[Manual Refine] ✗ Fallback failed: {e}")
                return JsonResponse({
                    "success": False,
                    "error": f"All refinement methods failed. Last error: {error_msg}"
                }, status=500)

        # Calculate statistics
        original_boxes_np = np.array(boxes)
        refined_boxes_np = np.array(box_out)

        # Count how many boxes changed
        boxes_changed = 0
        total_displacement = 0
        for i in range(len(original_boxes_np)):
            orig = original_boxes_np[i]
            refined = refined_boxes_np[i]
            displacement = np.sqrt(np.sum((orig - refined) ** 2))
            if displacement > 0.1:  # Threshold for "changed"
                boxes_changed += 1
            total_displacement += displacement

        avg_displacement = total_displacement / len(boxes)

        # Save refined boxes back to the .mat file
        print(f"[Manual Refine] Saving refined boxes back to {mat_path}...")

        # Reload and update the data structure
        data['data'][0, 0]['newBox'] = np.array(box_out)
        data['data'][0, 0]['order'] = np.array(box_order)
        data['data'][0, 0]['rBoundary'] = np.array([np.array(rb) for rb in rBoundary], dtype=object)

        # NEW: Store the NEXT pass number for the next refinement call
        # Store at top level to avoid structured array field issues
        # Pass sequence: 1 → 2 → 3 → 4 → 5 → 1
        # Pass 4: bathroom wall anchor (after bathroom snaps to LR in Pass 3)
        # Pass 5: living room expansion + coverage gap fill
        next_pass = current_pass + 1 if current_pass < 7 else 1
        data['refinement_pass'] = np.array([[next_pass]])

        # Save back to disk
        sio.savemat(mat_path, data)
        print(f"[Manual Refine] ✓ Saved refined floor plan to disk (next pass will be {next_pass}/7)")

        # Format data for frontend rendering (same format as AdjustGraph response)
        roomret = []
        for k in range(len(box_out)):
            room_label = mdul.room_label[int(room_types[k])][1]  # Get room type label
            order_idx = box_order[k][0] - 1 if isinstance(box_order[k], list) else k
            room_data = (box_out[k], [room_label], order_idx)
            roomret.append(room_data)

        # Format boundary as exterior string
        exterior = ""
        for i in range(len(boundary)):
            exterior += str(boundary[i][0]) + "," + str(boundary[i][1]) + " "
        exterior = exterior.strip()

        # Format door (first two boundary points)
        door = f"{boundary[0][0]},{boundary[0][1]},{boundary[1][0]},{boundary[1][1]}"

        # Return results with rendering data
        return JsonResponse({
            "success": True,
            "method": refinement_method,
            "threshold": threshold,
            "refinement_pass": current_pass,  # Which pass was just executed
            "next_pass": next_pass,  # Which pass will run next time
            "statistics": {
                "total_boxes": len(boxes),
                "boxes_changed": int(boxes_changed),
                "avg_displacement": float(avg_displacement),
                "max_displacement": float(np.max([np.sqrt(np.sum((original_boxes_np[i] - refined_boxes_np[i]) ** 2))
                                                  for i in range(len(boxes))]))
            },
            "message": f"Refinement pass {current_pass}/2 complete using {refinement_method} method",
            # Add rendering data
            "roomret": roomret,
            "exterior": exterior,
            "door": door,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.stdout = _old_stdout
        if current_pass == 1 or _last_refinement_log["userRoomID"] != userRoomID:
            _last_refinement_log["text"] = ""
        _last_refinement_log["userRoomID"] = userRoomID
        _last_refinement_log["text"] += _tee.getvalue()
        return JsonResponse({
            "success": False,
            "error": str(e)
        }, status=500)

    finally:
        # Restore stdout and accumulate the captured log.
        # Reset when starting a new cycle (Pass 1) or a different floor plan.
        sys.stdout = _old_stdout
        if current_pass == 1 or _last_refinement_log["userRoomID"] != userRoomID:
            _last_refinement_log["text"] = ""
        _last_refinement_log["userRoomID"] = userRoomID
        _last_refinement_log["text"] += _tee.getvalue()


def Download_Logs(request):
    """
    Serve the log from the most recent refinement call as a plain-text download.
    """
    import datetime
    log_text = _last_refinement_log.get("text") or "No refinement log yet."
    room_id = _last_refinement_log.get("userRoomID") or "unknown"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"refinement_log_{room_id}_{timestamp}.txt"

    response = HttpResponse(log_text, content_type="text/plain; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


def TransGraph_net(request):
    userInfo = request.GET.get("userInfo")
    testname = userInfo.split(',')[0]
    trainname = request.GET.get("roomID")
    mlresult = mltest.get_userinfo_net(testname, trainname)

    fp_end = mlresult[0]
    boxes_pred = mlresult[1]

    data_js = {}
    # fp_end  hsedge
    data_js["hsedge"] = (fp_end.get_triples(tensor=False)[:, [0, 2, 1]]).astype(float).tolist()

    # fp_end rmpos
    rooms = fp_end.get_rooms(tensor=False)
    room = rooms
    center = [[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in fp_end.data.box[:, :4]]

    

    # boxes_pred
    data_js["rmpos"] = []
    for k in range(len(center)):
        node = float(room[k]), mdul.room_label[int(room[k])][1], center[k][0], center[k][1]
        data_js["rmpos"].append(node)
    boxes_end = boxes_pred.tolist()
    data_js['roomret'] = []
    for k in range(len(room)):
        data = boxes_end[k], [mdul.room_label[int(room[k])][1]]
        data_js['roomret'].append(data)

    test_index = testNameList.index(testname.split(".")[0])
    data = test_data[test_index]
    ex = ""
    for i in range(len(data.boundary)):
        ex = ex + str(data.boundary[i][0]) + "," + str(data.boundary[i][1]) + " "
    data_js['exterior'] = ex
    x0, x1 = np.min(data.boundary[:, 0]), np.max(data.boundary[:, 0])
    y0, y1 = np.min(data.boundary[:, 1]), np.max(data.boundary[:, 1])
    data_js['bbxarea'] = float((x1 - x0) * (y1 - y0))
    return HttpResponse(json.dumps(data_js), content_type="application/json")


def GraphSearch(request):
    s=time.perf_counter()
    # Graph
    Searchtype = ["BedRoom", "Bathroom", "Kitchen", "Balcony", "Storage"]
    BedRoomlist = ["MasterRoom", "SecondRoom", "GuestRoom", "ChildRoom", "StudyRoom"]
    NewGraph = json.loads(request.GET.get("NewGraph"))
   
    testname = request.GET.get("userRoomID")
    newNode = NewGraph[0]
    newEdge = NewGraph[1]
    r_Num = np.zeros((1, 14)).tolist()
    r_Mask = np.zeros((1, 14)).tolist()
    r_Acc = np.zeros((1, 14)).tolist()
    r_Num[0][0] = 1
    r_Mask[0][0] = 1
    r_Acc[0][0] = 1

    for indx, rmname, x, y, scalesize in newNode:
        r_Num[0][mdul.vocab['object_name_to_idx'][rmname]] = r_Num[0][mdul.vocab['object_name_to_idx'][rmname]] + 1
        r_Mask[0][mdul.vocab['object_name_to_idx'][rmname]] = 1
        if rmname in BedRoomlist:
            r_Num[0][13] = r_Num[0][13] + 1
            r_Mask[0][13] = 1

    test_index = testNameList.index(testname.split(".")[0])
    topkList = []
    topkList.clear()
    data = test_data[test_index]
   
    Numrooms = json.loads(request.GET.get("Numrooms"))
    

    roomactarr = Numrooms[0]
    roomexaarr = Numrooms[1]
    roomnumarr = [int(x) for x in Numrooms[2]]
    test_data_topk=np.arange(0,74995)

    if np.sum(roomactarr) != 1 or np.sum(roomexaarr) != 1 or np.sum(roomnumarr) != 1:
        test_num = train_data_rNum[test_data_topk]
        # Number filter
     
        filter_func = get_filter_func(roomactarr, roomexaarr, roomnumarr)
        indices = np.where(list(map(filter_func, test_num)))
        # print("np.where(list(map(fil", test_num)
        indices = list(indices)
        test_data_topk = test_data_topk[indices[0]]

    test_num = train_data_eNum[test_data_topk]
    # Graph filter
    
    edgematrix = np.zeros((5, 5))
    for indx1, indx2 in newEdge:
        tmp1 = ""
        tmp2 = ""
        for indx, rmname, x, y, scalesize in newNode:
            if indx1 == indx:
                if rmname in BedRoomlist:
                    tmp1 = "BedRoom"
                else:
                    tmp1 = rmname
        for indx, rmname, x, y, scalesize in newNode:
            if indx2 == indx:
                if rmname in BedRoomlist:
                    tmp2 = "BedRoom"
                else:
                    tmp2 = rmname
        if tmp1 != "" and tmp2 != "":
            edgematrix[Searchtype.index(tmp1)][Searchtype.index(tmp2)] = edgematrix[Searchtype.index(tmp1)][
                                                                             Searchtype.index(tmp2)] + 1
            edgematrix[Searchtype.index(tmp2)][Searchtype.index(tmp1)] = edgematrix[Searchtype.index(tmp2)][
                                                                             Searchtype.index(tmp1)] + 1
    edge = edgematrix.reshape((1, 25))
    filter_graphfunc = filter_graph(edge)
    # rNum_list
    eNumData = []
   
    indices = np.where(list(map(filter_graphfunc, test_num)))
    indices = list(indices)

    topkList = []

    # FALLBACK: If hard filter returns no results, use similarity scoring
    if len(indices[0]) == 0:
        print("GraphSearch: Hard filter returned no results. Using similarity-based fallback.")

        # Get all candidates from test_data_topk for similarity scoring
        candidate_indices = test_data_topk

        # Compute TF distances for all candidates
        x, y = rt.compute_tf(data.boundary)
        y_sampled = rt.sample_tf(x, y, 1000)
        tf_distances = np.linalg.norm(y_sampled - tf_train[candidate_indices], axis=1)

        # Get room counts and edge data for similarity scoring
        candidate_rNum = train_data_rNum[candidate_indices]
        candidate_eNum = train_data_eNum[candidate_indices]

        # Compute similarity scores
        scores = compute_similarity_scores(
            candidate_rNum=candidate_rNum,
            candidate_eNum=candidate_eNum,
            requested_rNum=roomnumarr,
            requested_edge=edge.flatten(),
            tf_distances=tf_distances,
            mask=roomactarr,
            alpha=0.35,  # Room count weight
            beta=0.35,   # Edge structure weight
            gamma=0.30   # Boundary similarity weight
        )

        # Sort by similarity score (lower is better)
        sorted_indices = np.argsort(scores)
        topk = min(20, len(sorted_indices))

        for i in range(topk):
            candidate_idx = candidate_indices[sorted_indices[i]]
            floor_plan_name = str(trainNameList[int(candidate_idx)]) + ".png"

            # Calculate match percentage
            match_percentage = calculate_room_match_percentage(
                candidate_rNum[sorted_indices[i]],
                roomnumarr,
                roomactarr,
                roomexaarr
            )

            # Return as object with name and match percentage
            topkList.append({
                "name": floor_plan_name,
                "match": round(match_percentage, 1),
                "fallback": True  # Indicates this used fallback matching
            })
    else:
        # Original hard filter logic
        tf_trainsub = tf_train[test_data_topk[indices[0]]]
        re_data = train_data[test_data_topk[indices[0]]]
        test_data_tftopk = retrieve_bf(tf_trainsub, data, k=20)
        re_data = re_data[test_data_tftopk]

        if len(re_data) < 20:
            topk = len(re_data)
        else:
            topk = 20

        for i in range(topk):
            floor_plan_name = str(re_data[i].name) + ".png"

            # Calculate match percentage for hard filter results
            candidate_idx = trainNameList.index(re_data[i].name)
            match_percentage = calculate_room_match_percentage(
                train_data_rNum[candidate_idx],
                roomnumarr,
                roomactarr,
                roomexaarr
            )

            # Return as object with name and match percentage
            topkList.append({
                "name": floor_plan_name,
                "match": round(match_percentage, 1),
                "fallback": False  # Hard filter match
            })

    e = time.perf_counter()
    print('Graph Search time: %s Seconds' % (e - s))

    print("topkList", topkList)

    # Return both old format (for backward compatibility) and new format (with metadata)
    response_data = {
        "floorPlans": [item["name"] for item in topkList],  # Old format for UI compatibility
        "metadata": topkList  # New format with match scores
    }
    return HttpResponse(json.dumps(response_data), content_type="application/json")


def AutoAdjustGraph(request):
    """
    Automatically adjust the graph by adding missing rooms and removing excess rooms
    based on user requirements.
    """
    print("=== AutoAdjustGraph called ===")

    # Get data from request
    NewGraph = json.loads(request.GET.get("NewGraph"))
    Numrooms = json.loads(request.GET.get("Numrooms"))

    newNode = NewGraph[0]  # [[index, roomname, x, y, scalesize], ...]
    newEdge = NewGraph[1]  # [[u, v], ...]

    roomactarr = Numrooms[0]  # Which room types are active
    roomexaarr = Numrooms[1]  # Exact match requirements
    roomnumarr = [int(x) for x in Numrooms[2]]  # Requested room counts

    print(f"Current nodes: {len(newNode)}")
    print(f"Current edges: {len(newEdge)}")
    print(f"Room requirements: {roomnumarr}")

    # Room type mappings
    room_idx_to_name = {
        0: 'LivingRoom', 1: 'MasterRoom', 2: 'Kitchen', 3: 'Bathroom',
        4: 'DiningRoom', 5: 'ChildRoom', 6: 'StudyRoom', 7: 'SecondRoom',
        8: 'GuestRoom', 9: 'Balcony', 10: 'Entrance', 11: 'Storage', 12: 'Wall-in',
        13: 'MasterRoom'  # Index 13 = combined bedroom count, treat as MasterRoom
    }

    # Bedroom types (they can be grouped)
    bedroom_types = ['MasterRoom', 'ChildRoom', 'StudyRoom', 'SecondRoom', 'GuestRoom']

    # Count current rooms by type
    current_room_counts = {i: 0 for i in range(14)}
    for indx, rmname, x, y, scalesize in newNode:
        for room_idx, room_name in room_idx_to_name.items():
            if rmname == room_name or (room_name == 'MasterRoom' and rmname in bedroom_types):
                if room_name == 'MasterRoom' and rmname in bedroom_types:
                    # All bedrooms count towards bedroom count (index 1 and index 13)
                    current_room_counts[1] += 1
                    current_room_counts[13] += 1
                else:
                    current_room_counts[room_idx] += 1
                break

    print(f"Current room counts: {current_room_counts}")

    # Determine which rooms to add/remove
    rooms_to_add = []
    rooms_to_remove = []

    for room_idx in range(14):
        if not roomactarr[room_idx]:  # Skip inactive room types
            continue

        current_count = current_room_counts[room_idx]
        requested_count = roomnumarr[room_idx]

        if requested_count > current_count:
            # Need to add rooms
            num_to_add = requested_count - current_count
            room_name = room_idx_to_name[room_idx]
            for _ in range(num_to_add):
                rooms_to_add.append(room_name)
        elif requested_count < current_count:
            # Need to remove rooms (only if exact match required)
            if roomexaarr[room_idx]:
                num_to_remove = current_count - requested_count
                room_name = room_idx_to_name[room_idx]
                rooms_to_remove.append((room_name, num_to_remove))

    print(f"Rooms to add: {rooms_to_add}")
    print(f"Rooms to remove: {rooms_to_remove}")

    # Get boundary information to place new nodes within bounds
    # The boundary is stored in the test data
    hsname = None
    try:
        # Try to get hsname from cookie (if available from frontend)
        hsname = request.COOKIES.get('hsname', None)
        if hsname:
            testname = hsname.split('.')[0]
            test_index = testNameList.index(testname)
            data = test_data[test_index]
            boundary = data.boundary

            # Calculate boundary bounds
            min_x = float(np.min(boundary[:, 0]))
            max_x = float(np.max(boundary[:, 0]))
            min_y = float(np.min(boundary[:, 1]))
            max_y = float(np.max(boundary[:, 1]))

            # Add some padding so nodes don't sit exactly on the boundary
            padding = 20
            min_x += padding
            max_x -= padding
            min_y += padding
            max_y -= padding

            print(f"Boundary bounds: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}]")
        else:
            # Fallback: use existing nodes to estimate bounds
            if len(newNode) > 0:
                min_x = min([x for _, _, x, _, _ in newNode]) - 20
                max_x = max([x for _, _, x, _, _ in newNode]) + 20
                min_y = min([y for _, _, y, _, _ in newNode]) - 20
                max_y = max([y for _, _, y, _, _ in newNode]) + 20
            else:
                min_x, max_x = 50, 200
                min_y, max_y = 50, 200
    except Exception as e:
        print(f"Warning: Could not load boundary, using defaults: {e}")
        # Fallback: use existing nodes or defaults
        if len(newNode) > 0:
            min_x = min([x for _, _, x, _, _ in newNode]) - 20
            max_x = max([x for _, _, x, _, _ in newNode]) + 20
            min_y = min([y for _, _, y, _, _ in newNode]) - 20
            max_y = max([y for _, _, y, _, _ in newNode]) + 20
        else:
            min_x, max_x = 50, 200
            min_y, max_y = 50, 200

    # Add new rooms
    next_index = max([int(indx) for indx, _, _, _, _ in newNode]) + 1 if len(newNode) > 0 else 0

    # Minimum distance from existing nodes
    min_distance = 30  # pixels

    for room_name in rooms_to_add:
        # Try to find a position away from existing nodes
        max_attempts = 50
        best_x, best_y = None, None
        best_min_dist = 0

        for attempt in range(max_attempts):
            # Generate random position within boundary
            candidate_x = random.uniform(min_x, max_x)
            candidate_y = random.uniform(min_y, max_y)

            # Calculate minimum distance to any existing node
            if len(newNode) > 0:
                min_dist_to_existing = min(
                    ((candidate_x - x) ** 2 + (candidate_y - y) ** 2) ** 0.5
                    for _, _, x, y, _ in newNode
                )
            else:
                min_dist_to_existing = float('inf')

            # Keep track of the best position (furthest from existing nodes)
            if min_dist_to_existing > best_min_dist:
                best_min_dist = min_dist_to_existing
                best_x, best_y = candidate_x, candidate_y

            # If we found a position far enough away, use it
            if min_dist_to_existing >= min_distance:
                break

        new_x, new_y = best_x, best_y
        newNode.append([next_index, room_name, new_x, new_y, 1])  # Default scale = 1
        print(f"Added {room_name} at ({new_x:.1f}, {new_y:.1f}), min distance from existing: {best_min_dist:.1f}px")
        next_index += 1

    # Remove excess rooms (prefer nodes with fewer edges to minimize disconnection)
    for room_name, count_to_remove in rooms_to_remove:
        # Find all candidate nodes of this room type
        candidates = []
        for i, (indx, rmname, x, y, scalesize) in enumerate(newNode):
            if rmname == room_name or (room_name == 'MasterRoom' and rmname in bedroom_types):
                # Count how many edges this node has
                node_id = int(indx)
                edge_count = sum(1 for u, v in newEdge if int(u) == node_id or int(v) == node_id)
                candidates.append((edge_count, i, node_id, rmname))

        # Sort by edge count (ascending) - remove nodes with fewest edges first
        candidates.sort(key=lambda x: x[0])

        # Remove the nodes with fewest edges
        removed = 0
        for edge_count, list_idx, node_id, rmname in candidates:
            if removed >= count_to_remove:
                break

            # Find current index in newNode (indices shift as we remove)
            actual_idx = None
            for i, (indx, _, _, _, _) in enumerate(newNode):
                if int(indx) == node_id:
                    actual_idx = i
                    break

            if actual_idx is not None:
                print(f"Removing {rmname} (node {node_id}) with {edge_count} edge(s)")
                newNode.pop(actual_idx)
                # Remove all edges connected to this node
                newEdge = [[u, v] for u, v in newEdge if int(u) != node_id and int(v) != node_id]
                removed += 1

    # Note: New nodes are added without edges
    # Edges will be added later through edge prediction model
    # (Previously we auto-connected to nearest neighbor, but that's been removed)

    print(f"Adjusted nodes: {len(newNode)}")
    print(f"Adjusted edges: {len(newEdge)}")

    # Return the adjusted graph
    result = {
        "nodes": newNode,
        "edges": newEdge
    }

    return HttpResponse(json.dumps(result), content_type="application/json")


def retrieve_bf(tf_trainsub, datum, k=20):
    # compute tf for the data boundary
    x, y = rt.compute_tf(datum.boundary)
    y_sampled = rt.sample_tf(x, y, 1000)
    dist = np.linalg.norm(y_sampled - tf_trainsub, axis=1)
    if k > np.log2(len(tf_trainsub)):
        index = np.argsort(dist)[:k]
    else:
        index = np.argpartition(dist, k)[:k]
        index = index[np.argsort(dist[index])]
    return index


def Export_DXF(request):
    """
    Manual DXF export endpoint - triggered by button click.

    Exports the saved floor plan to DXF format in the network location.

    GET parameters:
        - userRoomID: Floor plan identifier (e.g., "14926")
        - scale: (optional) Scale factor (default from config)
        - filename: (optional) Custom filename (default: userRoomID.dxf)

    Returns:
        JSON with success status and file path
    """
    userRoomID = request.GET.get("userRoomID")
    custom_scale = request.GET.get("scale", None)
    custom_filename = request.GET.get("filename", None)

    if not userRoomID:
        return JsonResponse({
            "success": False,
            "error": "userRoomID parameter required"
        }, status=400)

    if not HAS_DXF_EXPORT:
        return JsonResponse({
            "success": False,
            "error": "DXF export not available. Please install ezdxf: pip install ezdxf"
        }, status=500)

    try:
        # Load the .mat file from static directory
        mat_path = f"./static/{userRoomID}.mat"
        if not os.path.exists(mat_path):
            return JsonResponse({
                "success": False,
                "error": f"Floor plan {userRoomID}.mat not found in static directory"
            }, status=404)

        print(f"\n[DXF Export] Loading {mat_path}...")
        data = sio.loadmat(mat_path)
        fp_data = data['data'][0, 0]

        # Debug: Check what fields are available
        print(f"[DXF Export Debug] Available fields: {fp_data.dtype.names}")

        # Debug: Check rType
        if 'rType' in fp_data.dtype.names:
            rType_data = fp_data['rType']
            print(f"[DXF Export Debug] rType shape: {rType_data.shape}, dtype: {rType_data.dtype}")
            print(f"[DXF Export Debug] rType values: {rType_data.flatten()}")
        else:
            print(f"[DXF Export Debug] rType NOT FOUND in .mat file!")

        # Debug: Check rBoundary
        if 'rBoundary' in fp_data.dtype.names:
            rBoundary_data = fp_data['rBoundary']
            print(f"[DXF Export Debug] rBoundary shape: {rBoundary_data.shape}, type: {type(rBoundary_data)}")
        else:
            print(f"[DXF Export Debug] rBoundary NOT FOUND in .mat file!")

        # Determine output filename
        if custom_filename:
            dxf_filename = custom_filename if custom_filename.endswith('.dxf') else f"{custom_filename}.dxf"
        else:
            dxf_filename = f"{userRoomID}.dxf"

        # Ensure network directory exists (create if needed)
        network_dir = DXF_SAVE_PATH
        try:
            os.makedirs(network_dir, exist_ok=True)
            print(f"[DXF Export] Network directory ready: {network_dir}")
        except Exception as e:
            return JsonResponse({
                "success": False,
                "error": f"Cannot access network path '{network_dir}': {str(e)}",
                "suggestion": "Check that N: drive is mapped and accessible"
            }, status=500)

        # Full output path
        dxf_path = os.path.join(network_dir, dxf_filename)

        # Determine scale
        scale = float(custom_scale) if custom_scale else DXF_SCALE

        print(f"[DXF Export] Exporting to {dxf_path}...")
        print(f"[DXF Export] Scale: {scale}, Wall thickness: {DXF_WALL_THICKNESS}")

        # Export to DXF
        success = save_floorplan_dxf(
            fp_data,
            dxf_path,
            scale=scale,
            wall_thickness=DXF_WALL_THICKNESS,
            include_labels=True,
            include_dimensions=False
        )

        if success:
            file_size = os.path.getsize(dxf_path) / 1024  # KB
            print(f"[DXF Export] ✓ Successfully exported {dxf_filename} ({file_size:.1f} KB)")

            # Count rooms for summary (flatten rType to get actual count)
            room_count = 0
            if 'rType' in fp_data.dtype.names:
                rType_array = np.array(fp_data['rType']).flatten()
                room_count = len(rType_array)
                print(f"[DXF Export] Detected {room_count} rooms")

            return JsonResponse({
                "success": True,
                "filename": dxf_filename,
                "path": dxf_path,
                "size_kb": round(file_size, 1),
                "scale": scale,
                "room_count": room_count,
                "message": f"DXF file saved successfully to network location"
            })
        else:
            return JsonResponse({
                "success": False,
                "error": "DXF export failed (check console for details)"
            }, status=500)

    except Exception as e:
        print(f"[DXF Export] ✗ Error: {e}")
        import traceback
        traceback.print_exc()

        return JsonResponse({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }, status=500)


def Log_Boundaries(request):
    """
    Boundary diagnostic endpoint - triggered by the Log Boundaries button.

    Logs all boundary point coordinates and checks each room box's position
    relative to the boundary polygon (extents check + Shapely escape area).

    GET parameters:
        - userRoomID: Floor plan identifier (e.g., "14926")

    Returns:
        JSON with boundary_points list and per-room relationship info.
        Full detail is also printed to the server console.
    """
    userRoomID = request.GET.get("userRoomID")
    if not userRoomID:
        return JsonResponse({"success": False, "error": "userRoomID parameter required"}, status=400)

    try:
        mat_path = f"./static/{userRoomID}.mat"
        if not os.path.exists(mat_path):
            return JsonResponse({"success": False,
                                 "error": f"Floor plan {userRoomID}.mat not found in static directory"}, status=404)

        print(f"\n[Log Boundaries] {'='*60}")
        print(f"[Log Boundaries] Floor plan: {userRoomID}")
        data = sio.loadmat(mat_path)
        fp_data = data['data'][0, 0]
        available_fields = fp_data.dtype.names
        print(f"[Log Boundaries] Available fields: {available_fields}")

        # --- Load boundary ---
        if 'boundary' not in available_fields:
            return JsonResponse({"success": False, "error": "No 'boundary' field in .mat file"})
        boundary = np.array(fp_data['boundary'])
        print(f"[Log Boundaries] Boundary shape: {boundary.shape}")

        # --- Load boxes and room types ---
        box_field = 'refineBox' if 'refineBox' in available_fields else 'gtBox'
        boxes_raw = np.array(fp_data[box_field]) if box_field in available_fields else None
        rtype_raw = np.array(fp_data['rType']).flatten() if 'rType' in available_fields else None

        room_type_names = {0: "LivingRoom", 1: "MasterRoom", 2: "Kitchen",
                           3: "Bathroom", 15: "FrontDoor"}
        direction_names  = {0: "right →", 1: "up ↑", 2: "left ←", 3: "down ↓"}

        # --- Parse boundary points ---
        coords = boundary[:, :2]
        boundary_points = []
        for idx in range(len(boundary)):
            pt = {"index": idx,
                  "x": round(float(coords[idx, 0]), 2),
                  "y": round(float(coords[idx, 1]), 2)}
            if boundary.shape[1] >= 3:
                d = int(boundary[idx, 2])
                pt["direction_code"] = d
                pt["direction"] = direction_names.get(d, str(d))
            if boundary.shape[1] >= 4:
                pt["is_new"] = int(boundary[idx, 3])
            boundary_points.append(pt)
            print(f"[Log Boundaries]   pt[{idx:2d}]  x={pt['x']:7.2f}  y={pt['y']:7.2f}"
                  + (f"  dir={pt['direction']}" if "direction" in pt else "")
                  + (f"  isNew={pt['is_new']}"  if "is_new"   in pt else ""))

        bnd_min_x = float(np.min(coords[:, 0]))
        bnd_max_x = float(np.max(coords[:, 0]))
        bnd_min_y = float(np.min(coords[:, 1]))
        bnd_max_y = float(np.max(coords[:, 1]))
        print(f"[Log Boundaries] Extents: x=[{bnd_min_x:.2f}, {bnd_max_x:.2f}]  "
              f"y=[{bnd_min_y:.2f}, {bnd_max_y:.2f}]")

        # --- Wall segments (point[i] → point[i+1]) ---
        n_pts = len(coords)
        wall_segments = []
        print(f"\n[Log Boundaries] Wall segments ({n_pts}):")
        for idx in range(n_pts):
            next_idx = (idx + 1) % n_pts
            x1w, y1w = float(coords[idx,      0]), float(coords[idx,      1])
            x2w, y2w = float(coords[next_idx, 0]), float(coords[next_idx, 1])
            length = round(float(np.sqrt((x2w - x1w)**2 + (y2w - y1w)**2)), 2)
            wall = {"index": idx,
                    "x1": round(x1w, 2), "y1": round(y1w, 2),
                    "x2": round(x2w, 2), "y2": round(y2w, 2),
                    "length": length}
            if boundary.shape[1] >= 3:
                d = int(boundary[idx, 2])
                wall["direction_code"] = d
                wall["direction"] = direction_names.get(d, str(d))
            wall_segments.append(wall)
            print(f"[Log Boundaries]   wall[{idx:2d}]  "
                  f"({x1w:.2f},{y1w:.2f}) → ({x2w:.2f},{y2w:.2f})  "
                  f"len={length:.2f}"
                  + (f"  dir={wall['direction']}" if "direction" in wall else ""))

        # --- Shapely polygon (optional, for escape area) ---
        bnd_poly = None
        has_shapely = False
        try:
            from shapely.geometry import Polygon as ShapelyPolygon
            from shapely.geometry import box as shapely_box
            bnd_poly = ShapelyPolygon(coords.tolist())
            if not bnd_poly.is_valid:
                bnd_poly = bnd_poly.buffer(0)
            has_shapely = True
        except ImportError:
            pass

        # --- Parse rooms ---
        rooms = []
        if boxes_raw is not None and rtype_raw is not None:
            print(f"\n[Log Boundaries] Rooms ({len(boxes_raw)}):")
            for i in range(len(boxes_raw)):
                row = boxes_raw[i]
                x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                rtype = int(rtype_raw[i]) if i < len(rtype_raw) else -1
                rname = room_type_names.get(rtype, f"Unknown({rtype})")
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                # Nearest boundary point
                dists      = np.sqrt((coords[:, 0] - cx)**2 + (coords[:, 1] - cy)**2)
                nearest_idx  = int(np.argmin(dists))
                nearest_dist = round(float(dists[nearest_idx]), 2)

                # Closest wall segment (point-to-segment distance from room center)
                min_wall_dist = float('inf')
                closest_wall_idx = 0
                for w in wall_segments:
                    wx1, wy1, wx2, wy2 = w['x1'], w['y1'], w['x2'], w['y2']
                    dx, dy = wx2 - wx1, wy2 - wy1
                    seg_len_sq = dx * dx + dy * dy
                    if seg_len_sq == 0:
                        wdist = float(np.sqrt((cx - wx1) ** 2 + (cy - wy1) ** 2))
                    else:
                        t = max(0.0, min(1.0, ((cx - wx1) * dx + (cy - wy1) * dy) / seg_len_sq))
                        proj_x = wx1 + t * dx
                        proj_y = wy1 + t * dy
                        wdist = float(np.sqrt((cx - proj_x) ** 2 + (cy - proj_y) ** 2))
                    if wdist < min_wall_dist:
                        min_wall_dist = wdist
                        closest_wall_idx = w['index']
                closest_wall = wall_segments[closest_wall_idx]
                closest_wall_name = f"Wall[{closest_wall_idx}]"
                if 'direction' in closest_wall:
                    closest_wall_name += f" {closest_wall['direction']}"
                closest_wall_dist = round(min_wall_dist, 2)

                room_info = {
                    "index": i, "type": rtype, "type_name": rname,
                    "x1": round(x1, 2), "y1": round(y1, 2),
                    "x2": round(x2, 2), "y2": round(y2, 2),
                    "center_x": round(cx, 2), "center_y": round(cy, 2),
                    "nearest_boundary_point_idx":  nearest_idx,
                    "nearest_boundary_point_dist": nearest_dist,
                    "closest_wall_idx":  closest_wall_idx,
                    "closest_wall_name": closest_wall_name,
                    "closest_wall_dist": closest_wall_dist,
                    "inside_boundary_extents": (x1 >= bnd_min_x and x2 <= bnd_max_x and
                                                y1 >= bnd_min_y and y2 <= bnd_max_y),
                }

                if has_shapely:
                    room_poly  = shapely_box(x1, y1, x2, y2)
                    outside    = room_poly.difference(bnd_poly)
                    escape_area = round(float(outside.area), 2) if not outside.is_empty else 0.0
                    room_info["escape_area_px2"]      = escape_area
                    room_info["fully_inside_boundary"] = escape_area < 0.01

                rooms.append(room_info)
                print(f"[Log Boundaries]   [{i}] {rname:12s} "
                      f"({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f})  "
                      f"nearest_bnd={nearest_idx} dist={nearest_dist:.2f}  "
                      f"closest_wall={closest_wall_name} wall_dist={closest_wall_dist:.2f}"
                      + (f"  escape={room_info['escape_area_px2']:.2f}px²" if has_shapely else "")
                      + ("  ⚠ OUTSIDE extents" if not room_info["inside_boundary_extents"] else ""))

        print(f"[Log Boundaries] {'='*60}\n")

        return JsonResponse({
            "success": True,
            "floor_plan_id": userRoomID,
            "boundary_point_count": len(boundary_points),
            "boundary_extents": {
                "x_min": round(bnd_min_x, 2), "x_max": round(bnd_max_x, 2),
                "y_min": round(bnd_min_y, 2), "y_max": round(bnd_max_y, 2),
            },
            "boundary_points": boundary_points,
            "wall_segments": wall_segments,
            "room_count": len(rooms),
            "rooms": rooms,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"success": False, "error": str(e),
                             "traceback": traceback.format_exc()}, status=500)


if __name__ == "__main__":
    pass