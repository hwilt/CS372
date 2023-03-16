import numpy as np

def average_columns(A):
    """
    Compute the average of each column of a matrix using only
    matrix multiplication and no loops

    Parameters
    ----------
    A: ndarray(M, N)
        A matrix

    Returns
    -------
    ndarray(N)
        An average across all columns
    """
    M = A.shape[0]
    N = A.shape[1]
    avg = np.zeros(N) # This is a dummy value
    ## TODO: Fill this in
    avg = np.sum(A, axis=0)/M
    return avg