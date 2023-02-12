import numpy as np
def has_freq(x, f, thresh):
    """
    Determine whether a frequency is contained in a signal
    
    Parameters
    ----------
    x: ndarray(N)
        The signal of interest
    f: int
        The frequency to check
    thresh: float
        If either the sine component or the cosine
        component are above this threshold, then consider
        the frequency f to exist
    
    Returns
    -------
    True if the frequency is in x, and False otherwise
    """
    N = len(x)
    n = np.arange(N)/N
    c = np.sum(x*np.cos(2*np.pi*f*n))
    ## TODO: Fill in sine part, and if either is > thresh, 
    ## then say it has the frequency
    s = np.sum(x*np.sin(2*np.pi*f*n))
    ret = False
    if np.abs(c) > thresh or np.abs(s) > thresh:
        ret = True
    return ret