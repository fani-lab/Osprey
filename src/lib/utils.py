import os
import numpy as np
from scipy import sparse

def save_sparse_csr(filename, array):
    """Save sparce csr matrix to .numpy file

    Args:
        filename (str): Name of the created file
        array (CSR Matrix): Sentence embeddings of the training and testing conversations
    """
    np.savez_compressed(filename, data=array.data, indices=array.indices, indptr =array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    """Load sparce csr matrix from .numpy file

    Args:
        filename (str): Name of the file

    Returns:
        CSR Matrix: Sentence embeddings of the training and testing conversations
    """
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])