import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import cv2
import copy
from model.utils import *

class FloorPlan():

    def __init__(self, data, rot=None, fliplr=False, index=None):
        self.data = copy.deepcopy(data)
        self.index = index  # Store index for fallback name generation

        # Compatibility layer: convert new format to old format if needed
        self._ensure_compatibility()

        ''' transform '''
        if rot is not None:
            theta = self._get_rot()
            self.data.gtBoxNew = align_box(self.data.gtBoxNew[:,[1,0,3,2]],theta,rot)[:,[1,0,3,2]]
            self.data.gtBoxNew = np.clip(self.data.gtBoxNew, 0, 255)  # Clip after rotation
            self.data.boundary[:,[1,0]] = align_points(self.data.boundary[:,[1,0]],theta,rot)
            self.data.boundary = np.clip(self.data.boundary, 0, 255)  # Clip after rotation
        
        if fliplr:
            self.data.gtBoxNew[:,[1,0,3,2]] = fliplr_box(self.data.gtBoxNew[:,[1,0,3,2]])
            self.data.gtBoxNew = np.clip(self.data.gtBoxNew, 0, 255)  # Clip after flip
            self.data.boundary[:,[1,0]] = fliplr_2D(self.data.boundary[:,[1,0]])
            self.data.boundary = np.clip(self.data.boundary, 0, 255)  # Clip after flip
    
    def _ensure_compatibility(self):
        """
        Convert new data format (box, edge) to old format (gtBoxNew, rType, rEdge) if needed.
        New format from 2.data_train_converted.py has:
        - box: (N, 5) array with [x0, y0, x1, y1, room_type]
        - edge: (E, 3) array with [room1, room2, edge_type]
        Old format expects:
        - gtBoxNew: (N, 4) array with [x0, y0, x1, y1]
        - rType: (N,) array with room types
        - rEdge: (E, 3) array with edges
        """
        if not hasattr(self.data, 'gtBoxNew') and hasattr(self.data, 'box'):
            # Extract box coordinates and room types from combined 'box' array
            box_data = self.data.box
            self.data.gtBoxNew = box_data[:, :4].astype(float)
            self.data.rType = box_data[:, 4].astype(int)
        
        if not hasattr(self.data, 'rEdge') and hasattr(self.data, 'edge'):
            # Rename 'edge' to 'rEdge'
            self.data.rEdge = self.data.edge.astype(int)
        
        # CRITICAL FIX: Validate and repair degenerate boxes
        self._validate_and_repair_boxes()

    def _validate_and_repair_boxes(self):
        """
        Check for and repair degenerate boxes (zero width/height).
        This can happen after rotation, clipping, or data preprocessing issues.
        """
        if not hasattr(self.data, 'gtBoxNew'):
            return
            
        boxes = self.data.gtBoxNew
        min_size = 2  # Minimum box size in pixels
        
        for i in range(len(boxes)):
            x0, y0, x1, y1 = boxes[i]
            
            # Check for zero or negative width/height
            if x1 <= x0:
                # Repair by expanding to minimum size
                center_x = (x0 + x1) / 2
                boxes[i, 0] = max(0, center_x - min_size / 2)
                boxes[i, 2] = min(255, center_x + min_size / 2)
                
            if y1 <= y0:
                # Repair by expanding to minimum size
                center_y = (y0 + y1) / 2
                boxes[i, 1] = max(0, center_y - min_size / 2)
                boxes[i, 3] = min(255, center_y + min_size / 2)
        
        self.data.gtBoxNew = boxes

    def _get_rot(self):
        door_line = self.data.boundary[:2, :2]  # [:,[1,0]]
        c = door_line.mean(0) - np.array([127.5,127.5])
        theta = np.arctan2(c[1], c[0]) + np.pi  # [-pi,pi]
        return theta

    def get_input_boundary(self, tensor=True):
        external = self.data.boundary[:, :2]
        door = self.data.boundary[:2, :2]

        boundary = np.zeros((128, 128), dtype=float)
        inside = np.zeros((128, 128), dtype=float)
        front = np.zeros((128, 128), dtype=float)

        # Clip boundary coordinates to valid range [0, 255] before dividing
        pts = np.clip(np.concatenate([external, external[:1]]), 0, 255) // 2
        pts_door = np.clip(door, 0, 255) // 2

        # Convert to int32 for OpenCV (required by fillPoly/polylines)
        pts = pts.astype(np.int32)
        pts_door = pts_door.astype(np.int32)

        # Safety checks
        if len(pts) < 3:
            raise ValueError(f"Invalid boundary polygon: need at least 3 points, got {len(pts)}")
        if len(pts_door) < 2:
            raise ValueError(f"Invalid door line: need at least 2 points, got {len(pts_door)}")

        cv2.fillPoly(inside, pts.reshape(1, -1, 2), 1.0)
        cv2.polylines(boundary, pts.reshape(1, -1, 2), True, 1.0, 3)
        cv2.polylines(boundary, pts_door.reshape(1, -1, 2), True, 0.5, 3)
        cv2.polylines(front, pts_door.reshape(1, -1, 2), True, 1.0, 3)

        input_image = np.stack([inside, boundary, front], -1)
        if tensor: input_image = torch.tensor(input_image).permute((2, 0, 1)).float()
        return input_image

    def get_inside_box(self, tensor=True):
        boundary = self.data.boundary[:, :2]

        # Clip boundary coordinates to valid range [0, 255]
        boundary = np.clip(boundary, 0, 255)
        
        # Get boundary extent in pixel coordinates (0-255)
        x0, x1 = np.min(boundary[:, 0]), np.max(boundary[:, 0])
        y0, y1 = np.min(boundary[:, 1]), np.max(boundary[:, 1])
        
        # Normalize to 0-1 range by dividing by 256 (not 255, to match coordinate system)
        box = np.array([[x0/256.0, y0/256.0, x1/256.0, y1/256.0]])
        box = np.clip(box, 0, 1)  # Ensure values stay in valid range
        
        if tensor: box = torch.tensor(box).float()
        return box

    def get_rooms(self, tensor=True):
        rooms = self.data.rType
        
        # Remap sparse data indices [0,1,2,3,15] to contiguous model indices [0,1,2,3,4]
        vocab = get_vocab()
        index_map = vocab['data_idx_to_model_idx']
        rooms = np.array([index_map.get(r, r) for r in rooms])
        
        if tensor: rooms = torch.tensor(rooms).long()
        return rooms

    def get_attributes(self, gsize=5, alevel=10, relative=True, tensor=True):
        boxes = self.data.gtBoxNew[:,[1,0,3,2]]
        boundary = self.data.boundary[:,:2]

        h, w = 256, 256
        if relative:
            x0, x1 = np.min(boundary[:, 0]), np.max(boundary[:, 0])+1
            y0, y1 = np.min(boundary[:, 1]), np.max(boundary[:, 1])+1
            h, w = int(y1 - y0), int(x1 - x0)
            boxes = boxes - np.array([y0, x0, y0, x0], dtype=float)

        boxes /= np.array([h, w, h, w])
        boxes[:, 2:] -= boxes[:, :2]  # y1,x1->h,w
        boxes[:, :2] += boxes[:, 2:] / 2  # y0,x0->yc,xc
        
        l = len(boxes)
        gbins = np.linspace(0,1,gsize+1) # [1,gsize]
        gbins[0],gbins[-1]=-np.inf,np.inf
        abins = np.linspace(0,1,alevel+1) # [1,gsize]
        abins[0],abins[-1]=-np.inf,np.inf
        
        attributes = np.zeros((l,gsize*gsize+alevel))
        # pos: xc*gsize+yc*gsize*gsize
        attributes[range(l),(np.digitize(boxes[:,0],gbins)-1)*gsize+np.digitize(boxes[:,1],gbins)-1]=1
        # area:(w*h)
        attributes[range(l),gsize*gsize+np.digitize(boxes[:,2:].prod(1),abins)-1]=1
        if tensor: attributes = torch.tensor(attributes).float()
        return attributes

    def get_triples(self, random=False, tensor=True):
        boxes = self.data.gtBoxNew[:, [1, 0, 3, 2]]
        vocab = get_vocab()

        triples = []
        
        # Handle edge data shape inconsistencies from MATLAB
        rEdge = self.data.rEdge
        
        # Convert to numpy array if needed
        if not isinstance(rEdge, np.ndarray):
            rEdge = np.array(rEdge)
        
        # Handle different edge array shapes
        if rEdge.ndim == 0 or (rEdge.ndim == 1 and len(rEdge) == 0):
            # Scalar or empty - no edges
            rEdge = np.array([]).reshape(0, 3)
        elif rEdge.ndim == 1:
            # 1D array - single edge, reshape to (1, 3)
            if len(rEdge) == 3:
                rEdge = rEdge.reshape(1, 3)
            else:
                # Invalid shape, skip edges
                rEdge = np.array([]).reshape(0, 3)
        # If ndim == 2, it's already correct shape
        
        # Get room type remapping (needed for embedding lookup)
        # Triples use room INDICES (not types), but we still need to ensure consistency
        # Note: u and v are indices into the room array, NOT room type indices
        # So we don't need to remap them - they're already correct array positions
        
        # add edge relation
        for u, v, _ in rEdge:
            # Todo: random order for edge
            # if random and np.random.random() > 0.5:
            #     u, v = v, u
            uy0, ux0, uy1, ux1 = boxes[u]
            vy0, vx0, vy1, vx1 = boxes[v]
            uc = (uy0 + uy1) / 2, (ux0 + ux1) / 2
            vc = (vy0 + vy1) / 2, (vx0 + vx1) / 2

            # surrounding/inside -> X four quadrants
            if ux0 < vx0 and ux1 > vx1 and uy0 < vy0 and uy1 > vy1:
                relation = 'surrounding'
            elif ux0 >= vx0 and ux1 <= vx1 and uy0 >= vy0 and uy1 <= vy1:
                relation = 'inside'
            else:
                relation = point_box_relation(uc, boxes[v])

            triples.append([u, vocab['pred_name_to_idx'][relation], v])

        # Handle empty triples (samples with no edges)
        if len(triples) == 0:
            triples = np.array([]).reshape(0, 3).astype(int)
        else:
            triples = np.array(triples, dtype=int)
        
        if tensor: triples = torch.tensor(triples).long()
        return triples

    def get_layout_image(self,tensor=True):
        # Use model index range for background (5 classes: 0-4)
        # Background and boundary both use index 5 (one beyond valid room indices) 
        # This will be ignored in CrossEntropyLoss with ignore_index=5
        vocab = get_vocab()
        num_classes = len(vocab['object_idx_to_name'])
        background_idx = num_classes  # One beyond valid indices (e.g., 5 for 5 classes)
        
        img = np.full((128,128), background_idx, dtype=np.uint8)
        boundary = self.data.boundary[:,:2]
        boundary = np.clip(boundary, 0, 255)  # Clip to valid range
        boundary = np.concatenate([boundary, boundary[:1]])

        cv2.fillPoly(img, (boundary//2).astype(np.int32).reshape(1, -1, 2), 0)

        order = (self.data.order-1).astype(int)
        rType = self.data.rType[order]
        rBox = self.data.gtBoxNew[order]
        
        # Remap sparse data indices to contiguous model indices
        index_map = vocab['data_idx_to_model_idx']
        rType = np.array([index_map.get(r, r) for r in rType])

        for i in range(len(rType)):
            t = rType[i]
            if t==0: continue
            b = rBox[i]//2
            img[b[1]:b[3],b[0]:b[2]]=t
        
        # Use same background_idx for boundary (will be ignored in loss)
        cv2.polylines(img, (boundary//2).astype(np.int32).reshape(1, -1, 2), True, background_idx)

        if tensor: img = torch.tensor(img).long()
        return img

    def get_boxes(self,relative=True,tensor=True):
        boxes = self.data.gtBoxNew[:, [1, 0, 3, 2]]
        boundary = self.data.boundary[:,:2]
        
        X,Y = np.linspace(0,1,256),np.linspace(0,1,256)

        if relative:
            x0, x1 = np.min(boundary[:, 0]), np.max(boundary[:, 0])+1
            y0, y1 = np.min(boundary[:, 1]), np.max(boundary[:, 1])+1
            h, w = int(y1 - y0), int(x1 - x0)
            X,Y = np.linspace(0,1,w),np.linspace(0,1,h)
            boxes = boxes-np.array([y0,x0,y0,x0])

        norm = lambda box:np.array([
            X[int(np.clip(box[1], 0, w-1))],
            Y[int(np.clip(box[0], 0, h-1))],
            X[int(np.clip(box[3]-1, 0, w-1))],
            Y[int(np.clip(box[2]-1, 0, h-1))]
        ])
        
        boxes = np.apply_along_axis(norm,1,boxes)
        boxes[:,2:]-=boxes[:,:2]
        boxes[:,:2]+=boxes[:,2:]/2
        if tensor: 
            boxes = torch.tensor(boxes).float()
        return boxes

    def get_inside_coords(self,size=(32,32),tensor=True):
        h,w = size
        X = np.linspace(0,1,w)
        Y = np.linspace(0,1,h)
        img = np.zeros(size)
        boundary = self.data.boundary[:,:2]
        boundary = np.clip(boundary, 0, 255)  # Clip to valid range
        boundary = boundary*np.array(size)//256

        cv2.fillPoly(img, boundary.astype(np.int32).reshape(1, -1, 2), 1)

        coords = np.where(img>0)
        coords = np.stack((X[coords[1]],Y[coords[0]]),1)
        if tensor: coords = torch.tensor(coords).unsqueeze(0).float()
        return coords

    def get_test_data(self, tensor=True):
        # Handle test data which may not have 'name' attribute
        name = self.data.name if hasattr(self.data, 'name') else f'sample_{self.index}'

        boundary = self.get_input_boundary(tensor=tensor)
        inside_box = self.get_inside_box(tensor=tensor)
        rooms = self.get_rooms(tensor=tensor)
        attrs = self.get_attributes(tensor=tensor)
        triples = self.get_triples(random=False, tensor=tensor)
        return boundary, inside_box, rooms, attrs, triples, name

    def get_train_data(self, tensor=True):
        # Handle test data which may not have 'name' attribute
        name = self.data.name if hasattr(self.data, 'name') else f'sample_{self.index}'

        boundary = self.get_input_boundary(tensor=tensor)
        inside_box = self.get_inside_box(tensor=tensor)
        rooms = self.get_rooms(tensor=tensor)
        attrs = self.get_attributes(tensor=tensor)
        triples = self.get_triples(random=False, tensor=tensor)

        # gt
        layout = self.get_layout_image(tensor=tensor)
        boxes = self.get_boxes(tensor=tensor)
        
        # constrains
        inside_coords = self.get_inside_coords(tensor=tensor)
        
        return boundary,inside_box,rooms,attrs,triples,layout,boxes,inside_coords,name

class FloorPlanDataset(Dataset):
    def __init__(self,data_path):
        self.data = sio.loadmat(data_path, squeeze_me=True, struct_as_record=False)['data']

        self.train = True if 'train' in data_path else False
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,i):
        if self.train:
            rot = np.random.randint(0,4)
            fliplr = np.random.random()>0.5
            fp = FloorPlan(self.data[i], rot=rot, fliplr=fliplr, index=i)
            return fp.get_train_data()
        else:
            fp = FloorPlan(self.data[i], index=i)
            return fp.get_train_data()
    
def floorplan_collate_fn(batch):
    all_boundary = []
    all_inside_box = []
    all_objs = []
    all_attrs = []
    all_triples = []

    all_layout = []
    all_boxes = []

    all_inside_coords = []
    
    all_obj_to_img = []
    all_triple_to_img = []

    all_name = []

    obj_offset = 0
    for i, (
        boundary,
        inside_box,
        rooms,
        attrs,
        triples,
        layout,
        boxes,
        inside_coords,
        name
        ) in enumerate(batch):
        if rooms.dim() == 0 or triples.dim() == 0:
            continue
        O, T = rooms.size(0), triples.size(0)

        all_boundary.append(boundary[None])
        all_inside_box.append(inside_box)

        all_objs.append(rooms)
        all_attrs.append(attrs)
        triples = triples.clone()
        triples[:, 0] += obj_offset
        triples[:, 2] += obj_offset
        all_triples.append(triples)

        all_layout.append(layout[None])
        all_boxes.append(boxes)
        all_inside_coords.append(inside_coords)

        all_name.append(name)
        
        all_obj_to_img.append(torch.LongTensor(O).fill_(i))
        all_triple_to_img.append(torch.LongTensor(T).fill_(i))

        obj_offset += O

    all_boundary = torch.cat(all_boundary)
    all_inside_box = torch.cat(all_inside_box)

    all_objs = torch.cat(all_objs)
    all_attrs = torch.cat(all_attrs)
    all_triples = torch.cat(all_triples)

    all_layout = torch.cat(all_layout)
    all_boxes = torch.cat(all_boxes)

    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)

    out = (
        all_boundary,
        all_inside_box, 
        all_objs,
        all_attrs, 
        all_triples,

        all_layout, 
        all_boxes, 

        all_inside_coords, 
        
        all_obj_to_img,
        all_triple_to_img,
        all_name
    )
    return out
    

def vis_fp(fp,size=(256,256)):
    cmap = get_color_map()

    img = np.full((256, 256, 4), 0, dtype=np.uint8)

    boundary = fp.data.boundary[:,:2]
    boundary = np.concatenate([boundary, boundary[:1]])
    door = fp.data.boundary[:2, :2]

    c = cmap[0].tolist()
    cv2.fillPoly(img, boundary.astype(np.int32).reshape(1, -1, 2), (c[0],c[1],c[2],255))

    order = (fp.data.order-1).astype(int)
    rType = fp.data.rType[order]
    rBox = fp.data.gtBoxNew[order]
    
    # Remap sparse data indices to contiguous model indices
    vocab = get_vocab()
    index_map = vocab['data_idx_to_model_idx']
    rType = np.array([index_map.get(r, r) for r in rType])

    for i in range(len(rType)):
        t = rType[i]
        if t==0: continue
        b = rBox[i]
        c = cmap[t].tolist()
        cv2.rectangle(img, (b[0],b[1]),(b[2]-1,b[3]-1), (c[0],c[1],c[2],255),-1)
        c = cmap[-1].tolist()
        cv2.rectangle(img, (b[0],b[1]),(b[2]-1,b[3]-1), (c[0],c[1],c[2],255),3)
        cv2.putText(img,f"{i}",((b[0]+b[2])//2,(b[1]+b[3])//2),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0,255))

    c = cmap[-3].tolist()
    cv2.polylines(img, boundary.astype(np.int32).reshape(1, -1, 2), True, (c[0],c[1],c[2],255),3)
    
    c = cmap[-2].tolist()
    cv2.polylines(img, door.astype(np.int32).reshape(1, -1, 2), True, (c[0],c[1],c[2],255),3)

    if size!=(256,256): return cv2.resize(img,size)
    return img

if __name__ == "__main__":
    pass
