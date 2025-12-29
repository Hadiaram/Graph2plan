from os import rmdir
import numpy as np
import pickle
import scipy.io as sio
from tqdm.auto import tqdm

data = pickle.load(open('./data/data_train_converted.pkl','rb'))['data']
names_train = open('./data/train.txt').read().split('\n')
n_train = len(names_train)

eNum = np.zeros((n_train,25),dtype='uint8')
for i in tqdm(range(n_train)):
    d = data[i]
    rType = d.box[:,-1].astype(int)

    # Handle edge array shape (might be 1D if only one edge)
    edges = d.edge
    if len(edges) == 0:
        # No edges - skip this floorplan
        continue
    if edges.ndim == 1:
        # Single edge - reshape to 2D
        edges = edges.reshape(1, -1)

    eType = rType[edges[:,:2].astype(int)]
    # classfication
    rMap = np.array([1,2,3,4,1,2,2,2,2,5,1,6,1,10,7,8,9,10])-1 # matlab to python
    edge = rMap[eType]
    reorder = np.array([0,1,3,2,4,5,6,7,8,9])  # Extended to handle all rMap outputs
    edge = reorder[edge]
    I = (edge[:,0]<=5)&(edge[:,0]>=1)&(edge[:,1]<=5)&(edge[:,1]>=1)
    edge = edge[I,:]-1 # matlab to python
    e = np.zeros((5,5),dtype='uint8')
    for j in range(len(edge)):
        e[edge[j,0],edge[j,1]] = e[edge[j,0],edge[j,1]]+1
        if edge[j,0] != edge[j,1]:
            e[edge[j,1],edge[j,0]] = e[edge[j,1],edge[j,0]]+1

    eNum[i] = e.reshape(-1)

pickle.dump({'eNum':eNum},open('./data/data_train_eNum.pkl','wb'))

