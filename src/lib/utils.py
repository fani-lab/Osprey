import os
import numpy as np
from scipy import sparse

def save_sparse_csr(filename, array):
    np.savez_compressed(filename, data=array.data, indices=array.indices, indptr =array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])