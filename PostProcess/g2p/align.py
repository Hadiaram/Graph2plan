import numpy as np
import os
import warnings

# Try to import MATLAB - make it optional
HAS_MATLAB = False
eng = None
try:
    import matlab
    import matlab.engine as engine
    eng = engine.start_matlab()
    eng.addpath(os.path.join(os.path.dirname(__file__),'matlab'),nargout=0)
    HAS_MATLAB = True
    print("MATLAB engine initialized successfully")
except ImportError:
    warnings.warn("MATLAB engine not available. Using Python fallback for alignment. "
                  "Install MATLAB and the MATLAB Python API for better results.", UserWarning)
except Exception as e:
    warnings.warn(f"Failed to initialize MATLAB engine: {e}. "
                  "Using Python fallback for alignment.", UserWarning)

GT_ThRESHOLD = 6
PRED_ThRESHOLD = 12
REFINE_ThRESHOLD = 18

def align_fp_python_fallback(boundary, boxes, types, edges, image, threshold, dtype=int):
    """
    Python fallback for align_fp when MATLAB is not available.
    Returns boxes without alignment, in a simple order based on position.

    Note: This is a simplified version. For best results, install MATLAB.
    """
    boxes = np.array(boxes, dtype=dtype)
    types = np.array(types, dtype=dtype)

    # Simple ordering: sort by y-coordinate (top to bottom), then x-coordinate (left to right)
    if len(boxes) > 0:
        centers = np.array([(boxes[i][0] + boxes[i][2]) / 2 for i in range(len(boxes))])
        y_coords = np.array([(boxes[i][1] + boxes[i][3]) / 2 for i in range(len(boxes))])
        order = np.argsort(y_coords * 1000 + centers)  # Sort by y first, then x
    else:
        order = np.array([], dtype=dtype)

    # Generate simple room boundaries (just bounding boxes)
    room_boundaries = []
    for box in boxes:
        if len(box) >= 4:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            # Create a rectangle as the room boundary
            boundary_poly = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
                [x1, y1]
            ], dtype=float)
            room_boundaries.append(boundary_poly)
        else:
            room_boundaries.append(np.array([[]], dtype=float))

    room_boundaries = np.array(room_boundaries, dtype=object)

    return boxes, order, room_boundaries

def align_fp(boundary, boxes, types, edges, image, threshold, dtype=int):
    """
    Aligns floorplan boxes using MATLAB if available, otherwise uses Python fallback.

    Args:
        boundary: Building boundary polygon
        boxes: Room bounding boxes
        types: Room types
        edges: Room connectivity graph
        image: Generated floorplan image
        threshold: Alignment threshold
        dtype: Data type for output arrays

    Returns:
        boxes_aligned: Aligned room boxes
        order: Drawing order for rooms
        room_boundaries: Room boundary polygons
    """
    boundary = np.array(boundary,dtype=int).tolist()
    boxes    = np.array(boxes,dtype=int).tolist()
    types    = np.array(types,dtype=int).tolist()
    edges    = np.array(edges,dtype=int).tolist()
    image    = np.array(image,dtype=int).tolist()

    if HAS_MATLAB and eng is not None:
        # Use MATLAB alignment
        boxes_aligned, order, room_boundaries = eng.align_fp(
            matlab.double(boundary),
            matlab.double(boxes),
            matlab.double(types),
            matlab.double(edges),
            matlab.double(image),
            threshold,False,nargout=3
        )

        boxes_aligned   = np.array(boxes_aligned,dtype=dtype)
        order           = np.array(order,dtype=dtype).reshape(-1)-1
        room_boundaries = np.array([np.array(rb,dtype=float) for rb in room_boundaries]) # poly with hole has value 'nan'
    else:
        # Use Python fallback
        boxes_aligned, order, room_boundaries = align_fp_python_fallback(
            boundary, boxes, types, edges, image, threshold, dtype
        )

    return boxes_aligned, order, room_boundaries

def align_fp_gt(boundary, boxes, types, edges, dtype=int):
    return align_fp(boundary, boxes, types, edges, [], GT_ThRESHOLD, dtype)

def align_fp_pred(boundary, boxes, types, edges, dtype=int):
    return align_fp(boundary, boxes, types, edges, [], PRED_ThRESHOLD, dtype)

def align_fp_refine(boundary, boxes, types, edges, image, dtype=int):
    return align_fp(boundary, boxes, types, edges, image, REFINE_ThRESHOLD, dtype)