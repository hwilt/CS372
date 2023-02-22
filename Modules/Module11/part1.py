import numpy as np
def get_window(x, win, hop, j):
    """
    Return the audio in the jth window
    Ex) First window (j = 0) x[0:win]
    Ex) Second window (j = 1) x[hop:hop+win]
    Ex) Third window (j = 2) x[2hop:2hop+win]
    
    Parameters
    ----------
    x: ndarray(N)
      Audio samples
    win: int
      Window length
    hop: int
      Hop length
    j: int 
      Window index
    """
    return x[j*hop:j*hop+win]