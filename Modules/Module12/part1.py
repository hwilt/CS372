import numpy as np
def max_freq(X):
    """
    Compute the index of the maximum amplitude frequency
    in complex DFT coefficients
    
    Parameters
    ----------
    X: ndarray(K, dtype=np.complex)
        The first K coefficients of the complex DFT
    
    Returns
    -------
    int: Index of the maximum amplitude frequency
    """
    return np.argmax(np.abs(X))