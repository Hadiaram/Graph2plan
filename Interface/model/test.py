from  model.floorplan import *
from  model.box_utils import *
from  model.model import Model
import os
from  model.utils import *

import Houseweb.views as vw

import numpy as np
import time
import math
import warnings

# Try to import MATLAB - make it optional
HAS_MATLAB = False
try:
    import matlab.engine #type: ignore
    HAS_MATLAB = True
except ImportError:
    warnings.warn("MATLAB engine not available in test.py. Using Python fallback.", UserWarning)

# Auto-detect device (GPU if available, otherwise CPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Model will use device: {DEVICE}")

global adjust,indxlist
adjust=False

def get_data(fp):
    batch = list(fp.get_test_data())
    batch[0] = batch[0].unsqueeze(0).to(DEVICE)
    batch[1] = batch[1].to(DEVICE)
    batch[2] = batch[2].to(DEVICE)
    batch[3] = batch[3].to(DEVICE)
    batch[4] = batch[4].to(DEVICE)
    return batch

def test(model,fp):
    with torch.no_grad():
        batch = get_data(fp)
        boundary,inside_box,rooms,attrs,triples = batch
        model_out = model(
            rooms, 
            triples, 
            boundary,
            obj_to_img = None,
            attributes = attrs,
            boxes_gt= None, 
            generate = True,
            refine = True,
            relative = True,
            inside_box=inside_box
        )
        boxes_pred,  gene_layout, boxes_refine= model_out
        boxes_pred = boxes_pred.detach()
        boxes_pred = centers_to_extents(boxes_pred)
        boxes_refine = boxes_refine.detach()
        boxes_refine = centers_to_extents(boxes_refine)
        gene_layout = gene_layout*boundary[:,:1]
        gene_preds = torch.argmax(gene_layout.softmax(1).detach(),dim=1)
        return boxes_pred.squeeze().cpu().numpy(),gene_preds.squeeze().cpu().double().numpy(),boxes_refine.squeeze().cpu().numpy()

def load_model():

    model = Model()
    model.to(DEVICE)
    model.load_state_dict(
        torch.load('./model/model.pth', map_location=DEVICE))
    model.eval()
    return model

def get_userinfo(userRoomID,adptRoomID):
    start = time.perf_counter()
    global model
    test_index = vw.testNameList.index(userRoomID.split(".")[0])
    test_data = vw.test_data[test_index]

    # boundary
    Boundary = test_data.boundary
    boundary=[[float(x),float(y),float(z),float(k)] for x,y,z,k in list(Boundary)]
    
    test_fp =FloorPlan(test_data)

    train_index = vw.trainNameList.index(adptRoomID.split(".")[0])
    train_data = vw.train_data[train_index]
    train_fp =FloorPlan(train_data,train=True)
    fp_end = test_fp.adapt_graph(train_fp)
    fp_end.adjust_graph()
    return fp_end


def calculate_average_room_sizes():
    """
    Calculate average room dimensions from training dataset.
    Returns dict: {room_type_id: (avg_width, avg_height)}
    """
    room_sizes = {}
    room_counts = {}

    # Iterate through all training data to collect room size statistics
    for train_item in vw.train_data:
        boxes = train_item.box  # [x1, y1, x2, y2, room_type]
        for box in boxes:
            x1, y1, x2, y2, room_type = box
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            room_type_id = int(room_type)

            if room_type_id not in room_sizes:
                room_sizes[room_type_id] = [0, 0]
                room_counts[room_type_id] = 0

            room_sizes[room_type_id][0] += width
            room_sizes[room_type_id][1] += height
            room_counts[room_type_id] += 1

    # Calculate averages
    avg_sizes = {}
    for room_type_id, (total_w, total_h) in room_sizes.items():
        count = room_counts[room_type_id]
        avg_sizes[room_type_id] = (total_w / count, total_h / count)

    return avg_sizes

def get_userinfo_adjust(userRoomID,adptRoomID,NewGraph):
    global adjust,indxlist
    test_index = vw.testNameList.index(userRoomID.split(".")[0])
    test_data = vw.test_data[test_index]
    # boundary
    Boundary = test_data.boundary
    boundary=[[float(x),float(y),float(z),float(k)] for x,y,z,k in list(Boundary)]

    test_fp =FloorPlan(test_data)

    train_index = vw.trainNameList.index(adptRoomID.split(".")[0])
    train_data = vw.train_data[train_index]
    train_fp =FloorPlan(train_data,train=True)
    fp_end = test_fp.adapt_graph(train_fp)
    fp_end.adjust_graph()


    newNode = NewGraph[0]
    newEdge = NewGraph[1]
    oldNode = NewGraph[2]
    
    temp = []
    for newindx, newrmname, newx, newy,scalesize in newNode:
        for type, oldrmname, oldx, oldy, oldindx in oldNode:
            if (int(newindx) == oldindx):
                tmp=int(newindx), (newx - oldx), ( newy- oldy),float(scalesize)
                temp.append(tmp)
    newbox=[]
    print(adjust)
    if adjust==True and vw.boxes_pred is not None:
        oldbox = []
        for i in range(len(vw.boxes_pred)):
            indxtmp=[vw.boxes_pred[i][0],vw.boxes_pred[i][1],vw.boxes_pred[i][2],vw.boxes_pred[i][3],vw.boxes_pred[i][0]]
            oldbox.append(indxtmp)
    else:
        indxlist=[]
        oldbox=fp_end.data.box.tolist()
        for i in range(len(oldbox)):
            indxlist.append([oldbox[i][4]])
        indxlist=np.array(indxlist)
        adjust=True
    oldbox=fp_end.data.box.tolist()

    # print("oldbox",oldbox)
    # print(oldbox,"oldbox")
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
            print(scalesize)
            newbox.append(tmpbox)

    fp_end.data.box=np.array(newbox)
    adjust_Edge=[]
    for u, v in newEdge:
        tmp=[index_mapping[int(u)],index_mapping[int(v)], 0]
        adjust_Edge.append(tmp)
    fp_end.data.edge=np.array(adjust_Edge)
    rNode = fp_end.get_rooms(tensor=False)

    rEdge = fp_end.get_triples(tensor=False)[:, [0, 2, 1]]
    Edge = [[float(u), float(v), float(type2)] for u, v, type2 in rEdge]

    # WORKAROUND: Use dataset-based box generation instead of model
    # Calculate average room sizes from training data (cached)
    if not hasattr(vw, '_avg_room_sizes'):
        print("📊 Calculating average room sizes from training dataset...")
        vw._avg_room_sizes = calculate_average_room_sizes()
        print(f"✅ Calculated averages for {len(vw._avg_room_sizes)} room types")

    # Calculate boundary area to scale rooms appropriately
    boundary_np = np.array(boundary)
    x_min, x_max = np.min(boundary_np[:, 0]), np.max(boundary_np[:, 0])
    y_min, y_max = np.min(boundary_np[:, 1]), np.max(boundary_np[:, 1])
    boundary_area = (x_max - x_min) * (y_max - y_min)

    # Generate boxes based on node positions and average room sizes
    avg_sizes = vw._avg_room_sizes
    boxes_pred = []
    total_room_area = 0

    # First pass: calculate total area with average sizes
    room_dims = []
    for i in range(len(fp_end.data.box)):
        room_type_id = int(fp_end.data.box[i][4])

        # Get average dimensions for this room type
        if room_type_id in avg_sizes:
            avg_w, avg_h = avg_sizes[room_type_id]
        else:
            # Fallback to a default size if room type not in training data
            avg_w, avg_h = 40.0, 40.0

        room_dims.append((avg_w, avg_h))
        total_room_area += avg_w * avg_h

    # Calculate scaling factor to fill ~70% of boundary area
    target_fill_ratio = 0.90
    if total_room_area > 0:
        area_scale = (boundary_area * target_fill_ratio) / total_room_area
        # Scale dimensions (not area), so take square root
        dim_scale = np.sqrt(area_scale)
        print(f"📐 Scaling rooms by {dim_scale:.2f}x to fill {target_fill_ratio*100:.0f}% of boundary")
    else:
        dim_scale = 1.0

    # Second pass: create boxes with scaled dimensions
    for i in range(len(fp_end.data.box)):
        avg_w, avg_h = room_dims[i]

        # Scale dimensions to better fill the boundary
        scaled_w = avg_w * dim_scale
        scaled_h = avg_h * dim_scale

        # Get center position from existing box
        x1, y1, x2, y2, _ = fp_end.data.box[i]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # Create box centered at this position with scaled dimensions
        new_x1 = cx - scaled_w / 2
        new_y1 = cy - scaled_h / 2
        new_x2 = cx + scaled_w / 2
        new_y2 = cy + scaled_h / 2

        boxes_pred.append([new_x1, new_y1, new_x2, new_y2])

    boxes_pred = np.array(boxes_pred)
    boxes_refeine = boxes_pred.copy()  # Use same for refinement

    # Create a simple gene_layout (just copy the original raster or create empty)
    # This is just for visualization, won't be perfect but will work
    gene_layout = fp_end.data.gene if hasattr(fp_end.data, 'gene') else np.zeros((128, 128))

    print(' ✅ Generated boxes using dataset averages (no model needed)')

    # NOTE: boxes_pred is already in 0-255 coordinate space (from fp_end.data.box),
    # so we DON'T multiply by 255 like the model path does (model outputs 0-1 normalized coords)

    fp_end.data.gene = gene_layout
    rBox = boxes_pred[:]
    Box = [[float(x), float(y), float(z), float(k)] for x, y, z, k in rBox]

    fp_end.data.boundary =np.array(boundary)
    fp_end.data.rType =np.array(rNode).astype(int)
    fp_end.data.refineBox=np.array(Box)
    fp_end.data.rEdge=np.array(Edge)

    startcom = time.perf_counter()
    if vw.engview is not None and HAS_MATLAB:
        # Use MATLAB alignment
        boundary_mat = matlab.double(boundary)
        rNode_mat = matlab.double(rNode.tolist())
        print("rNode.tolist()",rNode.tolist())
        Edge_mat = matlab.double(Edge)
        Box_mat = matlab.double(Box)
        gene_mat = matlab.double(np.array(fp_end.data.gene).tolist())

        box_refine = vw.engview.align_fp(boundary_mat, Box_mat, rNode_mat, Edge_mat,
                                         matlab.double(fp_end.data.gene.astype(float).copy().tolist()),
                                         18, False, nargout=3)
        box_out = box_refine[0]
        box_order = box_refine[1]
        rBoundary = box_refine[2]
    else:
        # Use Python fallback from views module
        print("Using Python fallback for alignment")
        box_out, box_order, rBoundary = vw._python_fallback_align(boundary, Box, rNode.tolist(), Edge, 18)

    endcom = time.perf_counter()
    print(' alignment compute time: %s Seconds' % (endcom - startcom))
    fp_end.data.newBox = np.array(box_out)
    fp_end.data.order = np.array(box_order)
    fp_end.data.rBoundary = [np.array(rb) for rb in rBoundary]
    return fp_end,box_out,box_order, gene_layout, boxes_refeine


def get_userinfo_net(userRoomID,adptRoomID):
    global model
    test_index = vw.testNameList.index(userRoomID.split(".")[0])
    test_data = vw.test_data[test_index]

    # boundary
    Boundary = test_data.boundary
    boundary = [[float(x), float(y), float(z), float(k)] for x, y, z, k in list(Boundary)]
    test_fp = FloorPlan(test_data)

    train_index = vw.trainNameList.index(adptRoomID.split(".")[0])
    train_data = vw.train_data[train_index]
    train_fp = FloorPlan(train_data, train=True)
    fp_end = test_fp.adapt_graph(train_fp)
    fp_end.adjust_graph()
    boxes_pred, gene_layout, boxes_refeine = test(model, fp_end)
    boxes_pred=boxes_pred*255
    for i in range(len(boxes_pred)):
        for j in range(len(boxes_pred[i])):
            boxes_pred[i][j]=float(boxes_pred[i][j])
    return fp_end,boxes_pred, gene_layout, boxes_refeine

if __name__ == "__main__":
    pass
