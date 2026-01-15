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
  

global adjust,indxlist
adjust=False

# Detect available device (GPU if available, otherwise CPU)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

def get_data(fp):
    print(f"   🔧 [TEST] get_data: Converting floor plan to batch tensors")
    batch = list(fp.get_test_data())
    print(f"      → batch elements: {len(batch)}")
    print(f"      → boundary shape: {batch[0].shape}, dtype: {batch[0].dtype}")
    print(f"      → inside_box shape: {batch[1].shape}")
    print(f"      → rooms shape: {batch[2].shape}, values: {batch[2]}")
    print(f"      → attrs shape: {batch[3].shape}")
    print(f"      → triples shape: {batch[4].shape}")
    batch[0] = batch[0].unsqueeze(0).to(DEVICE)
    batch[1] = batch[1].to(DEVICE)
    batch[2] = batch[2].to(DEVICE)
    batch[3] = batch[3].to(DEVICE)
    batch[4] = batch[4].to(DEVICE)
    print(f"      → Moved all tensors to {DEVICE}")
    return batch

def test(model,fp):
    print(f"\n   🧪 [TEST] test(): Running model inference")
    with torch.no_grad():
        batch = get_data(fp)
        boundary,inside_box,rooms,attrs,triples = batch
        
        print(f"   🔮 [TEST] Calling model forward pass...")
        print(f"      → rooms: {rooms.shape}")
        print(f"      → triples: {triples.shape}")
        print(f"      → boundary: {boundary.shape}")
        print(f"      → attrs: {attrs.shape}")
        print(f"      → inside_box: {inside_box.shape}")
        
        try:
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
            print(f"   ✅ [TEST] Model forward pass completed")
        except Exception as ex:
            print(f"   ❌ [TEST] Model forward pass FAILED: {type(ex).__name__}: {ex}")
            import traceback
            traceback.print_exc()
            raise
        
        boxes_pred,  gene_layout, boxes_refine= model_out
        print(f"      → boxes_pred shape: {boxes_pred.shape}, dtype: {boxes_pred.dtype}")
        print(f"      → gene_layout shape: {gene_layout.shape}, dtype: {gene_layout.dtype}")
        print(f"      → boxes_refine shape: {boxes_refine.shape}, dtype: {boxes_refine.dtype}")
        
        boxes_pred = boxes_pred.detach()
        boxes_pred = centers_to_extents(boxes_pred)
        boxes_refine = boxes_refine.detach()
        boxes_refine = centers_to_extents(boxes_refine)
        gene_layout = gene_layout*boundary[:,:1]
        gene_preds = torch.argmax(gene_layout.softmax(1).detach(),dim=1)
        
        print(f"   📦 [TEST] Post-processing complete:")
        print(f"      → boxes_pred (after centers_to_extents): {boxes_pred.shape}")
        print(f"      → gene_preds (after argmax): {gene_preds.shape}")
        print(f"      → boxes_refine (after centers_to_extents): {boxes_refine.shape}")
        
        return boxes_pred.squeeze().cpu().numpy(),gene_preds.squeeze().cpu().double().numpy(),boxes_refine.squeeze().cpu().numpy()

def load_model():

    model = Model()
    
    # Load model with appropriate device mapping first, then move to device
    if torch.cuda.is_available():
        model.load_state_dict(
            torch.load('./model/model.pth', map_location={'cuda:0': 'cuda:0'}))
        model.to(DEVICE)
    else:
        model.load_state_dict(
            torch.load('./model/model.pth', map_location='cpu'))
        # Don't need to move to device since model is already on CPU

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


def get_userinfo_adjust(userRoomID,adptRoomID,NewGraph):
    print(f"\n{'='*80}")
    print(f"🔬 [TEST] get_userinfo_adjust() called")
    print(f"{'='*80}")
    print(f"   → userRoomID: {userRoomID}")
    print(f"   → adptRoomID: {adptRoomID}")
    print(f"   → NewGraph: {len(NewGraph)} elements")
    
    global adjust,indxlist
    test_index = vw.testNameList.index(userRoomID.split(".")[0])
    test_data = vw.test_data[test_index]
    print(f"\n   📂 [TEST] Loading test data:")
    print(f"      → test_index: {test_index}")
    print(f"      → test_data type: {type(test_data)}")
    
    # boundary
    Boundary = test_data.boundary
    boundary=[[float(x),float(y),float(z),float(k)] for x,y,z,k in list(Boundary)]
    print(f"      → Boundary shape: {Boundary.shape}")
    print(f"      → Boundary points: {len(boundary)}")
    
    print(f"\n   🏗️ [TEST] Creating test FloorPlan...")
    test_fp =FloorPlan(test_data)
    print(f"      → test_fp created successfully")

    train_index = vw.trainNameList.index(adptRoomID.split(".")[0])
    train_data = vw.train_data[train_index]
    print(f"\n   📂 [TEST] Loading train data:")
    print(f"      → train_index: {train_index}")
    print(f"      → train_data type: {type(train_data)}")
    
    print(f"\n   🏗️ [TEST] Creating train FloorPlan...")
    train_fp =FloorPlan(train_data,train=True)
    print(f"      → train_fp created successfully")
    
    print(f"\n   🔀 [TEST] Adapting graph from train to test...")
    fp_end = test_fp.adapt_graph(train_fp)
    print(f"      → Graph adapted successfully")
    
    print(f"\n   ⚙️ [TEST] Adjusting graph...")
    fp_end.adjust_graph()
    print(f"      → Graph adjusted successfully")

    
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
    print(f"\n   📊 [TEST] Graph structure after user adjustments:")
    print(f"      → rNode (rooms): {rNode.shape}, values: {rNode}")

    rEdge = fp_end.get_triples(tensor=False)[:, [0, 2, 1]]
    Edge = [[float(u), float(v), float(type2)] for u, v, type2 in rEdge]
    print(f"      → rEdge (edges): {rEdge.shape}")
    print(f"      → Edge list: {len(Edge)} edges")
    print(f"      → fp_end.data.box shape: {fp_end.data.box.shape}")

    print(f"\n   🎯 [TEST] Loading model and running inference...")
    s=time.perf_counter()
    vw.loadModel()
    print(f"      → Model loaded, vw.model type: {type(vw.model)}")
    print(f"      → Model is on device: {next(vw.model.parameters()).device if hasattr(vw.model, 'parameters') else 'unknown'}")
    
    boxes_pred, gene_layout, boxes_refeine = test(vw.model, fp_end)

    e=time.perf_counter()
    print(f'\n   ⏱️ [TEST] Model inference time: {e - s:.3f} seconds')

    print(f"\n   🔢 [TEST] Scaling boxes by 255...")
    boxes_pred = boxes_pred * 255
    print(f"      → boxes_pred range: [{boxes_pred.min():.2f}, {boxes_pred.max():.2f}]")
    
    fp_end.data.gene = gene_layout
    rBox = boxes_pred[:]
    Box = [[float(x), float(y), float(z), float(k)] for x, y, z, k in rBox]
    print(f"      → Created Box list with {len(Box)} boxes")

    fp_end.data.boundary =np.array(boundary)
    fp_end.data.rType =np.array(rNode).astype(int)
    fp_end.data.refineBox=np.array(Box)
    fp_end.data.rEdge=np.array(Edge)
    print(f"      → Updated fp_end.data with predictions")

    print(f"\n   🔧 [TEST] Starting box alignment...")
    startcom = time.perf_counter()
    if vw.engview is not None and HAS_MATLAB:
        # Use MATLAB alignment
        print(f"      → Using MATLAB alignment (engview available)")
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
        print(f"      → MATLAB alignment completed")
    else:
        # Use Python fallback from views module
        print(f"      → Using Python fallback for alignment (MATLAB not available)")
        box_out, box_order, rBoundary = vw._python_fallback_align(boundary, Box, rNode.tolist(), Edge, 18)
        print(f"      → Python alignment completed")

    endcom = time.perf_counter()
    print(f'   ⏱️ [TEST] Alignment time: {endcom - startcom:.3f} seconds')
    print(f"      → box_out: {len(box_out)} boxes")
    print(f"      → box_order: {len(box_order)} entries")
    print(f"      → rBoundary: {len(rBoundary)} room boundaries")
    fp_end.data.newBox = np.array(box_out)
    fp_end.data.order = np.array(box_order)
    fp_end.data.rBoundary = [np.array(rb) for rb in rBoundary]
    
    print(f"\n   ✅ [TEST] get_userinfo_adjust() completed successfully")
    print(f"      → Returning: (fp_end, box_out, box_order, gene_layout, boxes_refeine)")
    print(f"{'='*80}\n")
    
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
