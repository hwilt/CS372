import numpy as np
def highpass_filter(x, num):
    """
    Parameters
    ----------
    x: ndarray(N)
        Audio samples
    num: int
        Number of times to convolve with the [1, -1] highpass filter
    
    
    Returns
    -------
    ndarray(N + 2*num-1)
        The result of applying the highpass filter
    """
    y = x
    ## TODO: Convolve with the highpass [1, -1] filter
    ## num times.  You can use the np.convolve command for now
    for i in range(num):
        y = np.convolve(y,[1,-1])
    return y
    