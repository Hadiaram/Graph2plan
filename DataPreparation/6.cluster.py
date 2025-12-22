import pickle
import numpy as np
import faiss #type: ignore
from tqdm.auto import tqdm

def sample_tf(x,y,ndim=1000):
    '''
    input: tf.x,tf.y, ndim
    return: n-dim tf values
    '''
    t = np.linspace(0,1,ndim)
    return np.piecewise(t,[t>=xx for xx in x],y)

tf_train = pickle.load(open('./data/trainTF.pkl','rb'))

tf = []
for i in tqdm(range(len(tf_train))):
    tf_i = tf_train[i]
    tf.append(sample_tf(tf_i['x'],tf_i['y']))

d = 1000
tf = np.array(tf).astype(np.float32)

# Auto-adjust cluster count based on dataset size
n_samples = len(tf)
ncentroids = min(1000, max(1, n_samples // 2))  # Use half the samples or 1000, whichever is smaller
nNN = min(1000, n_samples)  # Can't search for more neighbors than samples
niter = 200
verbose = True

print(f"Dataset size: {n_samples} samples")
print(f"Using {ncentroids} clusters and {nNN} nearest neighbors")

# Use GPU if available, otherwise CPU
try:
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
except:
    print("GPU not available, using CPU")
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=False)
kmeans.train(tf)
centroids = kmeans.centroids

index = faiss.IndexFlatL2(d)
index.add(tf)
D, I = index.search (kmeans.centroids, nNN)

np.save(f'./data/centroids_train.npy',centroids)
np.save(f'./data/clusters_train.npy',I)