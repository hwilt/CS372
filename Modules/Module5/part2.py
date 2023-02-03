import numpy as np
def cut_quiet(x, win, thresh):
    """
    Parameters
    ----------
    x: ndarray(N)
      Audio samples
    win: int
      Window in which to average intensity around each sample
    thresh: float
      Intensity threshold below which to cut samples, in dB
    """
    # Compute the average intensity in dB over a window
    L = x**2
    I = 10*np.log10(L) + 120
    I = (np.cumsum(I[win::]) - np.cumsum(I[0:-win]))/win
    # Chop down the audio at both ends so that y is parallel to I
    y = x[int(win/2):int(win/2)+len(I)]
    ## TODO: Only include samples in y that are strictly
    ## above the loudness threshold
    y = y[thresh < I]
    return y