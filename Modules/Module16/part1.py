import numpy as np
import matplotlib.pyplot as plt

def hann_window(N):
    """
    Create the Hann window 0.5*(1-cos(2pi*n/N))
    """
    return 0.5*(1 - np.cos(2*np.pi*np.arange(N)/N))

def specgram(x, w, h, win_fn = hann_window):
    """
    Compute the non-redundant amplitudes of the STFT
    Parameters
    ----------
    x: ndarray(N)
        Full audio clip of N samples
    w: int
        Window length
    h: int
        Hop length
    win_fn: int -> ndarray(N)
        Window function
    
    Returns
    -------
    ndarray(w, floor(w/2)+1, dtype=np.complex) STFT
    """
    N = len(x)
    nwin = int(np.ceil((N-w)/h))+1
    K = int(np.floor(w/2))+1
    # Make a 2D array
    # The rows correspond to frequency bins
    # The columns correspond to windows moved forward in time
    S = np.zeros((K, nwin))
    # Loop through all of the windows, and put the fourier
    # transform amplitudes of each window in its own column
    for j in range(nwin):
        # Pull out the audio in the jth window
        xj = x[h*j:h*j+w]
        # Zeropad if necessary
        if len(xj) < w:
            xj = np.concatenate((xj, np.zeros(w-len(xj))))
        # Apply window function
        xj = win_fn(w)*xj
        # Put the fourier transform into S
        sj = np.abs(np.fft.fft(xj))
        S[:, j] = sj[0:K]
    return S


def audio_offset_fn(x, w=2048, h=512):
    """
    Parameters
    ----------
    x: ndarray(N)
        Audio samples
    w: int
        Window length to use in spectrogram
    h: int
        Hop length to use in spectrogram
    
    Returns
    -------
    ndarray((N-win)/h+1)
        The audio offset function
    """
    eps = 1e-12
    S = specgram(x, w, h)
    S[S < eps] = eps
    Sdb = np.log10(S/eps)
    M = Sdb.shape[0] # How many rows I have (frequency indices)
    N = Sdb.shape[1] # How many columns I have (time window indices)
    diff = Sdb[:, 1::] - Sdb[:, 0:-1]
    diff[diff > 0] = 0
    novfn = np.sum(diff, axis=0)
    return novfn * -1