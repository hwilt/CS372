import numpy as np
def myfftconvolve(x, y):
    """
    Parameters
    ----------
    x: ndarray(N)
        Samples in the first array
    y: ndarray(M)
        Samples in the second array
    
    
    Returns
    -------
    ndarray(M+N-1)
        The result of the convolution
    """
    M = len(x)
    N = len(y)
    # Round M+N-1 up to the nearest power of 2
    K = int(2**np.ceil(np.log2(M+N-1)))
    xz = np.zeros(K)
    yz = np.zeros(K)
    ## TODO: Put x at the beginning of xz and y at the beginning of yz
    ## Then, take the DFT of xz and yz with np.fft.fft
    ## Then, multiply the DFTs and invert them with np.fft.ifft
    ## Finally, return the first M+N-1 samples of the result
    xz[:M] = x
    yz[:N] = y
    X = np.fft.fft(xz)
    Y = np.fft.fft(yz)
    Z = X*Y
    z = np.fft.ifft(Z)
    return z[:M+N-1]