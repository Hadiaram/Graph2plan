from django.shortcuts import render # type: ignore
from django.http import HttpResponse, JsonResponse # type: ignore
from django.conf import settings # type: ignore
import json
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

# Try to import MATLAB - make it optional
HAS_MATLAB = False
try:
    import matlab.engine #type: ignore
    HAS_MATLAB = True
except ImportError:
    warnings.warn("MATLAB engine not available. Some features may be limited.", UserWarning)

# Try to import DXF export
HAS_DXF_EXPORT = False
try:
    from Houseweb.dxf_export import save_floorplan_dxf, build_floorplan_dxf_bytes
    HAS_DXF_EXPORT = True
    print("[Init] DXF export module loaded successfully ✓")
except ImportError as e:
    print(f"[Init] DXF export not available: {e}")
    warnings.warn("DXF export not available. Install ezdxf: pip install ezdxf", UserWarning)

# DXF Export Configuration
ENABLE_AUTO_DXF_EXPORT = False  # Set to True to auto-save DXF on every save
DXF_SCALE = 1.0                 # Scale factor (1.0 = pixels, 0.01 = cm)
DXF_WALL_THICKNESS = 3.0        # Wall thickness in drawing units
DXF_SAVE_PATH = "/mnt/c/Users/hmbashir/source/DXF Floor Plans"

global test_data, test_data_topk, testNameList, trainNameList
global train_data, trainTF, train_data_eNum, train_data_rNum
global engview, model
global tf_train, centroids, clusters
global boxes_pred, indxlist
global last_fp_data, last_testname

# Initialize module-level variables
boxes_pred = None
indxlist = None
last_fp_data = None
last_testname = None


def _python_fallback_align(boundary, boxes, types, edges, threshold):
    """
    Python fallback for MATLAB align_fp when MATLAB is not available.
    Scales and translates boxes from their source bounding box to fill
    the target boundary, then clips to boundary limits.
    """
    boxes = np.array(boxes)
    types = np.array(types)
    boundary = np.array(boundary)

    # Target boundary extents
    bx_min = np.min(boundary[:, 0])
    bx_max = np.max(boundary[:, 0])
    by_min = np.min(boundary[:, 1])
    by_max = np.max(boundary[:, 1])
    margin = 5

    # Identify balconies (type 9 — allowed outside boundary)
    balcony_mask = np.array(
        [i < len(types) and int(types[i]) == 9 for i in range(len(boxes))],
        dtype=bool,
    )
    regular_mask = ~balcony_mask

    # Compute scale/offset to map the regular-room bounding box onto the
    # target boundary (non-uniform scale fills the boundary in both axes,
    # giving the CP-SAT optimizer a well-distributed starting point).
    sx = sy = 1.0
    offset_x = offset_y = 0.0
    if regular_mask.any():
        reg = boxes[regular_mask]
        src_x_min, src_x_max = np.min(reg[:, 0]), np.max(reg[:, 2])
        src_y_min, src_y_max = np.min(reg[:, 1]), np.max(reg[:, 3])
        src_w = src_x_max - src_x_min
        src_h = src_y_max - src_y_min

        tgt_x_min = bx_min + margin
        tgt_x_max = bx_max - margin
        tgt_y_min = by_min + margin
        tgt_y_max = by_max - margin

        sx = (tgt_x_max - tgt_x_min) / src_w if src_w > 0 else 1.0
        sy = (tgt_y_max - tgt_y_min) / src_h if src_h > 0 else 1.0
        offset_x = tgt_x_min - src_x_min * sx
        offset_y = tgt_y_min - src_y_min * sy

    print(f"[FALLBACK ALIGN] scale=({sx:.3f}, {sy:.3f})  offset=({offset_x:.1f}, {offset_y:.1f})")

    scaled_boxes = []
    for i, box in enumerate(boxes):
        if len(box) >= 4:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            if balcony_mask[i]:
                if x2 <= x1:
                    x2 = x1 + 10
                if y2 <= y1:
                    y2 = y1 + 10
                scaled_boxes.append([x1, y1, x2, y2])
            else:
                # Scale then translate
                nx1 = x1 * sx + offset_x
                ny1 = y1 * sy + offset_y
                nx2 = x2 * sx + offset_x
                ny2 = y2 * sy + offset_y

                # Clip to boundary with margin
                nx1 = max(bx_min + margin, min(nx1, bx_max - margin))
                nx2 = max(bx_min + margin, min(nx2, bx_max - margin))
                ny1 = max(by_min + margin, min(ny1, by_max - margin))
                ny2 = max(by_min + margin, min(ny2, by_max - margin))

                if nx2 <= nx1:
                    nx2 = nx1 + 10
                if ny2 <= ny1:
                    ny2 = ny1 + 10

                scaled_boxes.append([nx1, ny1, nx2, ny2])
        else:
            scaled_boxes.append(list(box))

    boxes = np.array(scaled_boxes)

    # Ordering: top-to-bottom, left-to-right (MATLAB 1-indexed)
    if len(boxes) > 0:
        centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
        centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
        order = np.argsort(centers_y * 1000 + centers_x) + 1
        order = order.reshape(-1, 1)
    else:
        order = np.array([[]], dtype=int)

    # Room boundary polygons (bounding rectangles)
    room_boundaries = []
    for box in boxes:
        if len(box) >= 4:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            room_boundaries.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])
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
    data_js["rmsize"] = [
        [[20 * math.sqrt((float(x2) - float(x1)) * (float(y2) - float(y1)) / float(area_))], [mdul.room_label[int(cate)][1]]]
        for
        x1, y1, x2, y2, cate in fp_end.data.box[:]]
    # fp_end rmpos

    rooms = fp_end.get_rooms(tensor=False)

    center = [[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in fp_end.data.box[:, :4]]

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
            # Clip to boundary limits
            x1 = max(xmin + margin, min(x1, xmax - margin))
            x2 = max(xmin + margin, min(x2, xmax - margin))
            y1 = max(ymin + margin, min(y1, ymax - margin))
            y2 = max(ymin + margin, min(y2, ymax - margin))
            # Ensure x2 > x1 and y2 > y1
            if x2 <= x1:
                x2 = x1 + 10
            if y2 <= y1:
                y2 = y1 + 10
            clipped_boxes_end.append([x1, y1, x2, y2])

    boxes_end = clipped_boxes_end
    print(f"   → Clipped {len(boxes_end)} boxes to boundary")

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

    # Store for OptimizeLayout endpoint
    global last_fp_data, last_testname
    last_fp_data = fp_end.data
    last_testname = testname

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
    
    mat_filename = "./static/" + testname.split(',')[0].split('.')[0] + ".mat"
    sio.savemat(mat_filename, {"data": fp_end.data})
    print(f"\n💾 [VIEWS] Saved floor plan data to {mat_filename}")

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


def OptimizeLayout(request):
    global last_fp_data, last_testname

    if last_fp_data is None:
        return HttpResponse(
            json.dumps({'error': 'No layout generated yet. Click Generate first.'}),
            content_type="application/json",
            status=400,
        )

    # Import solver (lives in PostProcess/optimizer/)
    import sys as _sys, os as _os
    _postprocess = _os.path.normpath(
        _os.path.join(_os.path.dirname(__file__), '..', '..', 'PostProcess'))
    if _postprocess not in _sys.path:
        _sys.path.insert(0, _postprocess)
    from optimizer.solver import optimize_layout, boxes_to_boundaries #type: ignore

    print("[OPTIMIZER] Running CP-SAT on current layout...")
    opt_boxes, opt_status = optimize_layout(
        last_fp_data.newBox,
        last_fp_data.rType,
        last_fp_data.rEdge,
        boundary=last_fp_data.boundary,
        timeout=15.0,
    )
    print(f"[OPTIMIZER] Status: {opt_status}")

    last_fp_data.newBox    = opt_boxes
    last_fp_data.rBoundary = boxes_to_boundaries(opt_boxes)
    last_fp_data = add_dw_fp(last_fp_data)

    # Build response in the same format as AdjustGraph
    external = np.asarray(last_fp_data.boundary)
    xmin, xmax = np.min(external[:, 0]), np.max(external[:, 0])
    ymin, ymax = np.min(external[:, 1]), np.max(external[:, 1])
    area_ = float((ymax - ymin) * (xmax - xmin)) or 1.0

    # Rooms sorted by area (largest first, same as AdjustGraph)
    K = len(opt_boxes)
    entries = []
    for i in range(K):
        box = [float(v) for v in opt_boxes[i]]
        rtype = int(last_fp_data.rType[i])
        room_name = mdul.room_label[rtype][1]
        area = (box[2] - box[0]) * (box[3] - box[1])
        entries.append((area, box, room_name, i))
    entries.sort(key=lambda e: e[0], reverse=True)

    data_js = {}
    data_js['roomret'] = [(box, [name], idx) for _, box, name, idx in entries]
    data_js['rmsize']  = [
        [[20 * math.sqrt(max(area, 0) / area_)], [name]]
        for area, _, name, _ in entries
    ]
    data_js['rmpos'] = []

    # Exterior boundary string
    ex = " ".join(f"{pt[0]},{pt[1]}" for pt in external)
    data_js['exterior'] = ex
    data_js['door'] = (f"{external[0][0]},{external[0][1]},"
                       f"{external[1][0]},{external[1][1]}")

    # Room boundary polygons
    data_js['indoor'] = []
    for rb in last_fp_data.rBoundary:
        if isinstance(rb, np.ndarray) and len(rb) > 0:
            data_js['indoor'].append(" ".join(f"{x},{y}" for x, y in rb))

    # Edges
    data_js['hsedge'] = last_fp_data.rEdge.astype(float).tolist()

    # Windows
    data_js['windows']     = []
    data_js['windowsline'] = []
    for indx, x, y, w, h, r in last_fp_data.windows:
        if w != 0:
            data_js['windows'].append([x + 2, y - 2, w - 2, 4])
            data_js['windowsline'].append([x + 2, y, w + x, y])
        if h != 0:
            data_js['windows'].append([x - 2, y, 4, h])
            data_js['windowsline'].append([x, y, x, h + y])

    # Save updated layout
    if last_testname:
        mat_filename = "./static/" + last_testname.split(',')[0].split('.')[0] + ".mat"
        sio.savemat(mat_filename, {"data": last_fp_data})

    print(f"[OPTIMIZER] Done — returning optimized layout.")
    return HttpResponse(json.dumps(data_js), content_type="application/json")


def ExpandLivingRoom(request):
    global last_fp_data, last_testname

    if last_fp_data is None:
        return HttpResponse(
            json.dumps({'error': 'No layout generated yet. Click Generate first.'}),
            content_type="application/json",
            status=400,
        )

    import sys as _sys, os as _os
    _postprocess = _os.path.normpath(
        _os.path.join(_os.path.dirname(__file__), '..', '..', 'PostProcess'))
    if _postprocess not in _sys.path:
        _sys.path.insert(0, _postprocess)
    from optimizer.solver import expand_living_room, boxes_to_boundaries #type: ignore

    print("[EXPAND LR] Running LivingRoom expansion pass...")
    exp_boxes = expand_living_room(
        last_fp_data.newBox,
        last_fp_data.rType,
        last_fp_data.boundary,
    )
    print("[EXPAND LR] Done.")

    last_fp_data.newBox    = exp_boxes
    last_fp_data.rBoundary = boxes_to_boundaries(exp_boxes)
    last_fp_data = add_dw_fp(last_fp_data)

    # Build response in the same format as OptimizeLayout / AdjustGraph
    external = np.asarray(last_fp_data.boundary)
    xmin, xmax = np.min(external[:, 0]), np.max(external[:, 0])
    ymin, ymax = np.min(external[:, 1]), np.max(external[:, 1])
    area_ = float((ymax - ymin) * (xmax - xmin)) or 1.0

    K = len(exp_boxes)
    entries = []
    for i in range(K):
        box = [float(v) for v in exp_boxes[i]]
        rtype = int(last_fp_data.rType[i])
        room_name = mdul.room_label[rtype][1]
        area = (box[2] - box[0]) * (box[3] - box[1])
        entries.append((area, box, room_name, i))
    entries.sort(key=lambda e: e[0], reverse=True)

    data_js = {}
    data_js['roomret'] = [(box, [name], idx) for _, box, name, idx in entries]
    data_js['rmsize']  = [
        [[20 * math.sqrt(max(area, 0) / area_)], [name]]
        for area, _, name, _ in entries
    ]
    data_js['rmpos'] = []

    ex = " ".join(f"{pt[0]},{pt[1]}" for pt in external)
    data_js['exterior'] = ex
    data_js['door'] = (f"{external[0][0]},{external[0][1]},"
                       f"{external[1][0]},{external[1][1]}")

    data_js['indoor'] = []
    for rb in last_fp_data.rBoundary:
        if isinstance(rb, np.ndarray) and len(rb) > 0:
            data_js['indoor'].append(" ".join(f"{x},{y}" for x, y in rb))

    data_js['hsedge'] = last_fp_data.rEdge.astype(float).tolist()

    data_js['windows']     = []
    data_js['windowsline'] = []
    for indx, x, y, w, h, r in last_fp_data.windows:
        if w != 0:
            data_js['windows'].append([x + 2, y - 2, w - 2, 4])
            data_js['windowsline'].append([x + 2, y, w + x, y])
        if h != 0:
            data_js['windows'].append([x - 2, y, 4, h])
            data_js['windowsline'].append([x, y, x, h + y])

    if last_testname:
        mat_filename = "./static/" + last_testname.split(',')[0].split('.')[0] + ".mat"
        sio.savemat(mat_filename, {"data": last_fp_data})

    return HttpResponse(json.dumps(data_js), content_type="application/json")


def AlignWalls(request):
    global last_fp_data, last_testname

    if last_fp_data is None:
        return HttpResponse(
            json.dumps({'error': 'No layout generated yet. Click Generate first.'}),
            content_type="application/json",
            status=400,
        )

    import sys as _sys, os as _os
    _postprocess = _os.path.normpath(
        _os.path.join(_os.path.dirname(__file__), '..', '..', 'PostProcess'))
    if _postprocess not in _sys.path:
        _sys.path.insert(0, _postprocess)
    from optimizer.solver import align_walls, boxes_to_boundaries  # type: ignore

    print("[ALIGN WALLS] Running wall alignment pass...")
    aligned_boxes = align_walls(last_fp_data.newBox, last_fp_data.rType)
    print("[ALIGN WALLS] Done.")

    last_fp_data.newBox    = aligned_boxes
    last_fp_data.rBoundary = boxes_to_boundaries(aligned_boxes)
    last_fp_data = add_dw_fp(last_fp_data)

    external = np.asarray(last_fp_data.boundary)
    xmin, xmax = np.min(external[:, 0]), np.max(external[:, 0])
    ymin, ymax = np.min(external[:, 1]), np.max(external[:, 1])
    area_ = float((ymax - ymin) * (xmax - xmin)) or 1.0

    K = len(aligned_boxes)
    entries = []
    for i in range(K):
        box = [float(v) for v in aligned_boxes[i]]
        rtype = int(last_fp_data.rType[i])
        room_name = mdul.room_label[rtype][1]
        area = (box[2] - box[0]) * (box[3] - box[1])
        entries.append((area, box, room_name, i))
    entries.sort(key=lambda e: e[0], reverse=True)

    data_js = {}
    data_js['roomret'] = [(box, [name], idx) for _, box, name, idx in entries]
    data_js['rmsize']  = [
        [[20 * math.sqrt(max(area, 0) / area_)], [name]]
        for area, _, name, _ in entries
    ]
    data_js['rmpos'] = []

    ex = " ".join(f"{pt[0]},{pt[1]}" for pt in external)
    data_js['exterior'] = ex
    data_js['door'] = (f"{external[0][0]},{external[0][1]},"
                       f"{external[1][0]},{external[1][1]}")

    data_js['indoor'] = []
    for rb in last_fp_data.rBoundary:
        if isinstance(rb, np.ndarray) and len(rb) > 0:
            data_js['indoor'].append(" ".join(f"{x},{y}" for x, y in rb))

    data_js['hsedge'] = last_fp_data.rEdge.astype(float).tolist()

    data_js['windows']     = []
    data_js['windowsline'] = []
    for indx, x, y, w, h, r in last_fp_data.windows:
        if w != 0:
            data_js['windows'].append([x + 2, y - 2, w - 2, 4])
            data_js['windowsline'].append([x + 2, y, w + x, y])
        if h != 0:
            data_js['windows'].append([x - 2, y, 4, h])
            data_js['windowsline'].append([x, y, x, h + y])

    if last_testname:
        mat_filename = "./static/" + last_testname.split(',')[0].split('.')[0] + ".mat"
        sio.savemat(mat_filename, {"data": last_fp_data})

    return HttpResponse(json.dumps(data_js), content_type="application/json")


def FillWallGaps(request):
    global last_fp_data, last_testname

    if last_fp_data is None:
        return HttpResponse(
            json.dumps({'error': 'No layout generated yet. Click Generate first.'}),
            content_type="application/json",
            status=400,
        )

    import sys as _sys, os as _os
    _postprocess = _os.path.normpath(
        _os.path.join(_os.path.dirname(__file__), '..', '..', 'PostProcess'))
    if _postprocess not in _sys.path:
        _sys.path.insert(0, _postprocess)
    from optimizer.solver import fill_wall_gaps, boxes_to_boundaries #type: ignore

    print("[FILL GAPS] Running wall gap fill pass...")
    filled_boxes = fill_wall_gaps(
        last_fp_data.newBox,
        last_fp_data.rType,
        last_fp_data.boundary,
    )
    print("[FILL GAPS] Done.")

    last_fp_data.newBox    = filled_boxes
    last_fp_data.rBoundary = boxes_to_boundaries(filled_boxes)
    last_fp_data = add_dw_fp(last_fp_data)

    # Build response in the same format as OptimizeLayout / AdjustGraph
    external = np.asarray(last_fp_data.boundary)
    xmin, xmax = np.min(external[:, 0]), np.max(external[:, 0])
    ymin, ymax = np.min(external[:, 1]), np.max(external[:, 1])
    area_ = float((ymax - ymin) * (xmax - xmin)) or 1.0

    K = len(filled_boxes)
    entries = []
    for i in range(K):
        box = [float(v) for v in filled_boxes[i]]
        rtype = int(last_fp_data.rType[i])
        room_name = mdul.room_label[rtype][1]
        area = (box[2] - box[0]) * (box[3] - box[1])
        entries.append((area, box, room_name, i))
    entries.sort(key=lambda e: e[0], reverse=True)

    data_js = {}
    data_js['roomret'] = [(box, [name], idx) for _, box, name, idx in entries]
    data_js['rmsize']  = [
        [[20 * math.sqrt(max(area, 0) / area_)], [name]]
        for area, _, name, _ in entries
    ]
    data_js['rmpos'] = []

    ex = " ".join(f"{pt[0]},{pt[1]}" for pt in external)
    data_js['exterior'] = ex
    data_js['door'] = (f"{external[0][0]},{external[0][1]},"
                       f"{external[1][0]},{external[1][1]}")

    data_js['indoor'] = []
    for rb in last_fp_data.rBoundary:
        if isinstance(rb, np.ndarray) and len(rb) > 0:
            data_js['indoor'].append(" ".join(f"{x},{y}" for x, y in rb))

    data_js['hsedge'] = last_fp_data.rEdge.astype(float).tolist()

    data_js['windows']     = []
    data_js['windowsline'] = []
    for indx, x, y, w, h, r in last_fp_data.windows:
        if w != 0:
            data_js['windows'].append([x + 2, y - 2, w - 2, 4])
            data_js['windowsline'].append([x + 2, y, w + x, y])
        if h != 0:
            data_js['windows'].append([x - 2, y, 4, h])
            data_js['windowsline'].append([x, y, x, h + y])

    if last_testname:
        mat_filename = "./static/" + last_testname.split(',')[0].split('.')[0] + ".mat"
        sio.savemat(mat_filename, {"data": last_fp_data})

    return HttpResponse(json.dumps(data_js), content_type="application/json")


def SnapRooms(request):
    global last_fp_data, last_testname

    if last_fp_data is None:
        return HttpResponse(
            json.dumps({'error': 'No layout generated yet. Click Generate first.'}),
            content_type="application/json",
            status=400,
        )

    import sys as _sys, os as _os
    _postprocess = _os.path.normpath(
        _os.path.join(_os.path.dirname(__file__), '..', '..', 'PostProcess'))
    if _postprocess not in _sys.path:
        _sys.path.insert(0, _postprocess)
    from optimizer.solver import snap_single_edge_rooms, boxes_to_boundaries #type: ignore

    print("[SNAP ROOMS] Running snap pass...")
    snapped_boxes = snap_single_edge_rooms(
        last_fp_data.newBox,
        last_fp_data.rType,
        last_fp_data.boundary,
        max_gap=80.0,
    )
    print("[SNAP ROOMS] Done.")

    last_fp_data.newBox    = snapped_boxes
    last_fp_data.rBoundary = boxes_to_boundaries(snapped_boxes)
    last_fp_data = add_dw_fp(last_fp_data)

    external = np.asarray(last_fp_data.boundary)
    xmin, xmax = np.min(external[:, 0]), np.max(external[:, 0])
    ymin, ymax = np.min(external[:, 1]), np.max(external[:, 1])
    area_ = float((ymax - ymin) * (xmax - xmin)) or 1.0

    K = len(snapped_boxes)
    entries = []
    for i in range(K):
        box = [float(v) for v in snapped_boxes[i]]
        rtype = int(last_fp_data.rType[i])
        room_name = mdul.room_label[rtype][1]
        area = (box[2] - box[0]) * (box[3] - box[1])
        entries.append((area, box, room_name, i))
    entries.sort(key=lambda e: e[0], reverse=True)

    data_js = {}
    data_js['roomret'] = [(box, [name], idx) for _, box, name, idx in entries]
    data_js['rmsize']  = [
        [[20 * math.sqrt(max(area, 0) / area_)], [name]]
        for area, _, name, _ in entries
    ]
    data_js['rmpos'] = []

    ex = " ".join(f"{pt[0]},{pt[1]}" for pt in external)
    data_js['exterior'] = ex
    data_js['door'] = (f"{external[0][0]},{external[0][1]},"
                       f"{external[1][0]},{external[1][1]}")

    data_js['indoor'] = []
    for rb in last_fp_data.rBoundary:
        if isinstance(rb, np.ndarray) and len(rb) > 0:
            data_js['indoor'].append(" ".join(f"{x},{y}" for x, y in rb))

    data_js['hsedge'] = last_fp_data.rEdge.astype(float).tolist()

    data_js['windows']     = []
    data_js['windowsline'] = []
    for indx, x, y, w, h, r in last_fp_data.windows:
        if w != 0:
            data_js['windows'].append([x + 2, y - 2, w - 2, 4])
            data_js['windowsline'].append([x + 2, y, w + x, y])
        if h != 0:
            data_js['windows'].append([x - 2, y, 4, h])
            data_js['windowsline'].append([x, y, x, h + y])

    if last_testname:
        mat_filename = "./static/" + last_testname.split(',')[0].split('.')[0] + ".mat"
        sio.savemat(mat_filename, {"data": last_fp_data})

    return HttpResponse(json.dumps(data_js), content_type="application/json")


def FillLivingRoom(request):
    global last_fp_data, last_testname

    if last_fp_data is None:
        return HttpResponse(
            json.dumps({'error': 'No layout generated yet. Click Generate first.'}),
            content_type="application/json",
            status=400,
        )

    import sys as _sys, os as _os
    _postprocess = _os.path.normpath(
        _os.path.join(_os.path.dirname(__file__), '..', '..', 'PostProcess'))
    if _postprocess not in _sys.path:
        _sys.path.insert(0, _postprocess)
    from optimizer.solver import fill_living_room, boxes_to_boundaries #type: ignore

    print("[FILL LR] Running living room fill pass...")
    filled_boxes = fill_living_room(
        last_fp_data.newBox,
        last_fp_data.rType,
        last_fp_data.boundary,
    )
    print("[FILL LR] Done.")

    last_fp_data.newBox    = filled_boxes
    last_fp_data.rBoundary = boxes_to_boundaries(filled_boxes)
    last_fp_data = add_dw_fp(last_fp_data)

    external = np.asarray(last_fp_data.boundary)
    xmin, xmax = np.min(external[:, 0]), np.max(external[:, 0])
    ymin, ymax = np.min(external[:, 1]), np.max(external[:, 1])
    area_ = float((ymax - ymin) * (xmax - xmin)) or 1.0

    K = len(filled_boxes)
    entries = []
    for i in range(K):
        box = [float(v) for v in filled_boxes[i]]
        rtype = int(last_fp_data.rType[i])
        room_name = mdul.room_label[rtype][1]
        area = (box[2] - box[0]) * (box[3] - box[1])
        entries.append((area, box, room_name, i))
    entries.sort(key=lambda e: e[0], reverse=True)

    data_js = {}
    data_js['roomret'] = [(box, [name], idx) for _, box, name, idx in entries]
    data_js['rmsize']  = [
        [[20 * math.sqrt(max(area, 0) / area_)], [name]]
        for area, _, name, _ in entries
    ]
    data_js['rmpos'] = []

    ex = " ".join(f"{pt[0]},{pt[1]}" for pt in external)
    data_js['exterior'] = ex
    data_js['door'] = (f"{external[0][0]},{external[0][1]},"
                       f"{external[1][0]},{external[1][1]}")

    data_js['indoor'] = []
    for rb in last_fp_data.rBoundary:
        if isinstance(rb, np.ndarray) and len(rb) > 0:
            data_js['indoor'].append(" ".join(f"{x},{y}" for x, y in rb))

    data_js['hsedge'] = last_fp_data.rEdge.astype(float).tolist()

    data_js['windows']     = []
    data_js['windowsline'] = []
    for indx, x, y, w, h, r in last_fp_data.windows:
        if w != 0:
            data_js['windows'].append([x + 2, y - 2, w - 2, 4])
            data_js['windowsline'].append([x + 2, y, w + x, y])
        if h != 0:
            data_js['windows'].append([x - 2, y, 4, h])
            data_js['windowsline'].append([x, y, x, h + y])

    if last_testname:
        mat_filename = "./static/" + last_testname.split(',')[0].split('.')[0] + ".mat"
        sio.savemat(mat_filename, {"data": last_fp_data})

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

    if engview is not None and HAS_MATLAB:
        # Use MATLAB alignment
        boundary_mat = matlab.double(boundary)
        rType_mat = matlab.double(rType.tolist())
        Edge_mat = matlab.double(Edge)
        Box_mat = matlab.double(Box)
        box_refine = engview.align_fp(boundary_mat, Box_mat,  rType_mat, Edge_mat, 18, False, nargout=3)
        box_out = box_refine[0]
        box_order = box_refine[1]
        rBoundary = box_refine[2]
    else:
        # Use Python fallback
        box_out, box_order, rBoundary = _python_fallback_align(boundary, Box, rType.tolist(), Edge, 18)
    fp_end.data.newBox = np.array(box_out)
    fp_end.data.order = np.array(box_order)
    fp_end.data.rBoundary = [np.array(rb) for rb in rBoundary]
    fp_end.data = add_dw_fp(fp_end.data)
    sio.savemat("./static/" + userRoomID + ".mat", {"data": fp_end.data})
    flag=1
    return HttpResponse(json.dumps(flag), content_type="application/json")


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
    Manual DXF export endpoint — streams the DXF file directly to the browser
    as a download (no server-side file path needed).
    """
    global last_fp_data, last_testname

    try:
        if not HAS_DXF_EXPORT:
            return JsonResponse({
                "success": False,
                "error": "DXF export not available. Please install ezdxf: pip install ezdxf"
            }, status=500)

        if last_fp_data is None or not last_testname:
            return JsonResponse({
                "success": False,
                "error": "No layout generated yet. Click Generate first."
            }, status=400)

        base_id = last_testname.split(',')[0].split('.')[0]
        custom_scale = request.GET.get("scale", None)
        custom_filename = request.GET.get("filename", None)

        dxf_filename = custom_filename if custom_filename else f"{base_id}.dxf"
        if not dxf_filename.endswith('.dxf'):
            dxf_filename += '.dxf'

        scale = float(custom_scale) if custom_scale else DXF_SCALE

        print(f"[DXF Export] Building DXF for {dxf_filename} (scale={scale})...")
        dxf_bytes = build_floorplan_dxf_bytes(
            last_fp_data,
            scale=scale,
            wall_thickness=DXF_WALL_THICKNESS,
            include_labels=True,
            include_dimensions=False,
        )

        room_count = len(np.array(last_fp_data.rType).flatten()) if hasattr(last_fp_data, 'rType') else 0
        print(f"[DXF Export] ✓ Built {dxf_filename} ({len(dxf_bytes)//1024:.1f} KB, {room_count} rooms)")

        response = HttpResponse(dxf_bytes, content_type='application/dxf')
        response['Content-Disposition'] = f'attachment; filename="{dxf_filename}"'
        response['X-DXF-Filename'] = dxf_filename
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"success": False, "error": str(e)}, status=500)


def Log_Boundaries(request):
    """
    Boundary diagnostic endpoint — triggered by the Log Boundary button.
    Logs all boundary point coordinates and checks each room box's position
    relative to the boundary polygon (extents check + Shapely escape area).
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

        if 'boundary' not in available_fields:
            return JsonResponse({"success": False, "error": "No 'boundary' field in .mat file"})
        boundary = np.array(fp_data['boundary'])
        print(f"[Log Boundaries] Boundary shape: {boundary.shape}")

        box_field = 'newBox' if 'newBox' in available_fields else (
                    'refineBox' if 'refineBox' in available_fields else 'gtBox')
        boxes_raw = np.array(fp_data[box_field]) if box_field in available_fields else None
        rtype_raw = np.array(fp_data['rType']).flatten() if 'rType' in available_fields else None

        room_type_names = {
            0: 'LivingRoom', 1: 'MasterRoom', 2: 'Kitchen', 3: 'Bathroom',
            4: 'DiningRoom', 5: 'ChildRoom', 6: 'StudyRoom', 7: 'SecondRoom',
            8: 'GuestRoom', 9: 'Balcony', 10: 'Entrance', 11: 'Storage',
            12: 'Wall-in', 13: 'External', 14: 'ExteriorWall', 15: 'FrontDoor',
        }
        direction_names = {0: "right →", 1: "up ↑", 2: "left ←", 3: "down ↓"}

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
        print(f"[Log Boundaries] Extents: x=[{bnd_min_x:.2f}, {bnd_max_x:.2f}]  y=[{bnd_min_y:.2f}, {bnd_max_y:.2f}]")

        n_pts = len(coords)
        wall_segments = []
        for idx in range(n_pts):
            next_idx = (idx + 1) % n_pts
            x1w, y1w = float(coords[idx, 0]),      float(coords[idx, 1])
            x2w, y2w = float(coords[next_idx, 0]), float(coords[next_idx, 1])
            length = round(float(np.sqrt((x2w - x1w)**2 + (y2w - y1w)**2)), 2)
            wall = {"index": idx, "x1": round(x1w, 2), "y1": round(y1w, 2),
                    "x2": round(x2w, 2), "y2": round(y2w, 2), "length": length}
            if boundary.shape[1] >= 3:
                d = int(boundary[idx, 2])
                wall["direction_code"] = d
                wall["direction"] = direction_names.get(d, str(d))
            wall_segments.append(wall)
            print(f"[Log Boundaries]   wall[{idx:2d}]  ({x1w:.2f},{y1w:.2f}) → ({x2w:.2f},{y2w:.2f})  len={length:.2f}"
                  + (f"  dir={wall['direction']}" if "direction" in wall else ""))

        bnd_poly = None
        has_shapely = False
        try:
            from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box #type: ignore
            bnd_poly = ShapelyPolygon(coords.tolist())
            if not bnd_poly.is_valid:
                bnd_poly = bnd_poly.buffer(0)
            has_shapely = True
        except ImportError:
            pass

        rooms = []
        if boxes_raw is not None and rtype_raw is not None:
            for i in range(len(boxes_raw)):
                row = boxes_raw[i]
                x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                rtype = int(rtype_raw[i]) if i < len(rtype_raw) else -1
                rname = room_type_names.get(rtype, f"Unknown({rtype})")
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                dists = np.sqrt((coords[:, 0] - cx)**2 + (coords[:, 1] - cy)**2)
                nearest_idx  = int(np.argmin(dists))
                nearest_dist = round(float(dists[nearest_idx]), 2)

                min_wall_dist = float('inf')
                closest_wall_idx = 0
                for w in wall_segments:
                    wx1, wy1, wx2, wy2 = w['x1'], w['y1'], w['x2'], w['y2']
                    dx, dy = wx2 - wx1, wy2 - wy1
                    seg_len_sq = dx * dx + dy * dy
                    if seg_len_sq == 0:
                        wdist = float(np.sqrt((cx - wx1)**2 + (cy - wy1)**2))
                    else:
                        t = max(0.0, min(1.0, ((cx - wx1) * dx + (cy - wy1) * dy) / seg_len_sq))
                        wdist = float(np.sqrt((cx - (wx1 + t * dx))**2 + (cy - (wy1 + t * dy))**2))
                    if wdist < min_wall_dist:
                        min_wall_dist = wdist
                        closest_wall_idx = w['index']

                closest_wall = wall_segments[closest_wall_idx]
                closest_wall_name = f"Wall[{closest_wall_idx}]"
                if 'direction' in closest_wall:
                    closest_wall_name += f" {closest_wall['direction']}"

                room_info = {
                    "index": i, "type": rtype, "type_name": rname,
                    "x1": round(x1, 2), "y1": round(y1, 2),
                    "x2": round(x2, 2), "y2": round(y2, 2),
                    "center_x": round(cx, 2), "center_y": round(cy, 2),
                    "nearest_boundary_point_idx":  nearest_idx,
                    "nearest_boundary_point_dist": nearest_dist,
                    "closest_wall_idx":  closest_wall_idx,
                    "closest_wall_name": closest_wall_name,
                    "closest_wall_dist": round(min_wall_dist, 2),
                    "inside_boundary_extents": (x1 >= bnd_min_x and x2 <= bnd_max_x and
                                                y1 >= bnd_min_y and y2 <= bnd_max_y),
                }
                if has_shapely:
                    from shapely.geometry import box as shapely_box #type: ignore
                    room_poly  = shapely_box(x1, y1, x2, y2)
                    outside    = room_poly.difference(bnd_poly)
                    escape_area = round(float(outside.area), 2) if not outside.is_empty else 0.0
                    room_info["escape_area_px2"]       = escape_area
                    room_info["fully_inside_boundary"] = escape_area < 0.01
                rooms.append(room_info)
                print(f"[Log Boundaries]   [{i}] {rname:12s} ({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f})"
                      + (f"  escape={room_info['escape_area_px2']:.2f}px²" if has_shapely else "")
                      + ("  ⚠ OUTSIDE extents" if not room_info["inside_boundary_extents"] else ""))

        print(f"[Log Boundaries] {'='*60}\n")
        return JsonResponse({
            "success": True,
            "floor_plan_id": userRoomID,
            "boundary_point_count": len(boundary_points),
            "boundary_extents": {"x_min": round(bnd_min_x, 2), "x_max": round(bnd_max_x, 2),
                                 "y_min": round(bnd_min_y, 2), "y_max": round(bnd_max_y, 2)},
            "boundary_points": boundary_points,
            "wall_segments":   wall_segments,
            "room_count": len(rooms),
            "rooms": rooms,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"success": False, "error": str(e),
                             "traceback": traceback.format_exc()}, status=500)


def Log_Graph(request):
    """Return the room graph (rEdge + rType + boxes) as JSON for the Log Graph button."""
    global last_fp_data
    try:
        if last_fp_data is None:
            return JsonResponse({"success": False, "error": "No layout generated yet."})

        rType_raw = getattr(last_fp_data, 'rType', None)
        if rType_raw is None:
            return JsonResponse({"success": False, "error": "rType not found on fp_data."})
        rType = np.array(rType_raw).flatten().astype(int)

        def room_name(t):
            return mdul.room_label[t][1] if 0 <= t < len(mdul.room_label) else f"Unknown({t})"

        rooms = [{"index": int(i), "type": int(t), "name": room_name(int(t))}
                 for i, t in enumerate(rType)]

        rEdge_raw = getattr(last_fp_data, 'rEdge', None)
        if rEdge_raw is None:
            return JsonResponse({"success": False,
                                 "error": "rEdge not found on fp_data.", "rooms": rooms})
        edges_arr = np.array(rEdge_raw)
        edges = []
        for row in edges_arr:
            u, v = int(row[0]), int(row[1])
            edge_type = int(row[2]) if len(row) > 2 else -1
            edges.append({
                "u": u, "u_name": room_name(int(rType[u])) if u < len(rType) else "?",
                "v": v, "v_name": room_name(int(rType[v])) if v < len(rType) else "?",
                "edge_type": edge_type,
            })

        newBox_raw = getattr(last_fp_data, 'newBox', None)
        boxes = []
        if newBox_raw is not None:
            for i, b in enumerate(newBox_raw):
                arr = np.array(b, dtype=float).flatten()[:4]
                boxes.append({"index": i, "x0": round(float(arr[0])), "y0": round(float(arr[1])),
                              "x1": round(float(arr[2])), "y1": round(float(arr[3]))})

        return JsonResponse({
            "success": True,
            "room_count": len(rooms),
            "rooms": rooms,
            "edge_count": len(edges),
            "edges": edges,
            "boxes": boxes,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"success": False, "error": str(e)}, status=500)


def FixRooms(request):
    global last_fp_data, last_testname

    if last_fp_data is None:
        return HttpResponse(
            json.dumps({'error': 'No layout generated yet. Click Generate first.'}),
            content_type="application/json",
            status=400,
        )

    import sys as _sys, os as _os
    _postprocess = _os.path.normpath(
        _os.path.join(_os.path.dirname(__file__), '..', '..', 'PostProcess'))
    if _postprocess not in _sys.path:
        _sys.path.insert(0, _postprocess)
    import importlib, optimizer.postprocess as _pp_mod
    importlib.reload(_pp_mod)
    fix_room_connectivity = _pp_mod.fix_room_connectivity
    from optimizer.solver import boxes_to_boundaries  # type: ignore

    try:
        print("[FIX ROOMS] Running fix rooms pass...")
        fixed_boxes = fix_room_connectivity(
            last_fp_data.newBox,
            last_fp_data.rType,
            last_fp_data.rEdge,
            last_fp_data.boundary,
        )
        print("[FIX ROOMS] Done.")

        last_fp_data.newBox    = fixed_boxes
        last_fp_data.rBoundary = boxes_to_boundaries(fixed_boxes)
        last_fp_data = add_dw_fp(last_fp_data)

        external = np.asarray(last_fp_data.boundary)
        xmin, xmax = np.min(external[:, 0]), np.max(external[:, 0])
        ymin, ymax = np.min(external[:, 1]), np.max(external[:, 1])
        area_ = float((ymax - ymin) * (xmax - xmin)) or 1.0

        K = len(fixed_boxes)
        entries = []
        for i in range(K):
            box = [float(v) for v in fixed_boxes[i]]
            rtype = int(last_fp_data.rType[i])
            room_name = mdul.room_label[rtype][1]
            area = (box[2] - box[0]) * (box[3] - box[1])
            entries.append((area, box, room_name, i))
        entries.sort(key=lambda e: e[0], reverse=True)

        data_js = {}
        data_js['roomret'] = [(box, [name], idx) for _, box, name, idx in entries]
        data_js['rmsize']  = [
            [[20 * math.sqrt(max(area, 0) / area_)], [name]]
            for area, _, name, _ in entries
        ]
        data_js['rmpos'] = []

        ex = " ".join(f"{pt[0]},{pt[1]}" for pt in external)
        data_js['exterior'] = ex
        data_js['door'] = (f"{external[0][0]},{external[0][1]},"
                           f"{external[1][0]},{external[1][1]}")

        data_js['indoor'] = []
        for rb in last_fp_data.rBoundary:
            if isinstance(rb, np.ndarray) and len(rb) > 0:
                data_js['indoor'].append(" ".join(f"{x},{y}" for x, y in rb))

        data_js['hsedge'] = last_fp_data.rEdge.astype(float).tolist()

        data_js['windows']     = []
        data_js['windowsline'] = []
        for indx, x, y, w, h, r in last_fp_data.windows:
            if w != 0:
                data_js['windows'].append([x + 2, y - 2, w - 2, 4])
                data_js['windowsline'].append([x + 2, y, w + x, y])
            if h != 0:
                data_js['windows'].append([x - 2, y, 4, h])
                data_js['windowsline'].append([x, y, x, h + y])

        if last_testname:
            mat_filename = "./static/" + last_testname.split(',')[0].split('.')[0] + ".mat"
            sio.savemat(mat_filename, {"data": last_fp_data})

        return HttpResponse(json.dumps(data_js), content_type="application/json")

    except Exception as _e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"success": False, "error": str(_e)}, status=500)

if __name__ == "__main__":
    pass