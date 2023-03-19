import numpy as np
import matplotlib.pyplot as plt

def get_mel_filterbank(K, win_length, sr, min_freq, max_freq, n_bins):
    """
    Compute a mel-spaced filterbank and place it in a matrix
    
    Parameters
    ----------
    win_length: int
        Window length
    K: int
        Number of frequency bins
    sr: int
        The sample rate used to generate sdb
    min_freq: int
        The center of the minimum mel bin, in hz
    max_freq: int
        The center of the maximum mel bin, in hz
    n_bins: int
        The number of mel bins to use
    
    Returns
    -------
    ndarray(n_bins, n_win)
        The mel-spaced spectrogram
    """
    # Space bin centers exponentially between min_freq and max_freq
    bins = np.logspace(np.log10(min_freq), np.log10(max_freq), n_bins+2)*win_length/sr
    bins = np.array(np.round(bins), dtype=int)
    Mel = np.zeros((n_bins, K))
    for i in range(n_bins):
        i1 = bins[i]
        i2 = bins[i+1]
        if i1 == i2:
            i2 += 1
        i3 = bins[i+2]
        if i3 <= i2:
            i3 = i2+1
        tri = np.zeros(K)
        ## TODO: Create a triangle in the tri array which
        ## goes from 0 to 1 between index i1 and i2
        ## and from 1 to 0 between index i2 and i3.
        ## Then, place this triangle at row i of the Mel matrix
        # hint use np.linspace
        tri[i1:i2] = np.linspace(0, 1, i2-i1)
        tri[i2:i3] = np.linspace(1, 0, i3-i2)
        Mel[i] = tri
        
    return Mel

def get_mel_specgram(x, sr, win_length, hop_size, min_freq, max_freq, n_bins):
    S = specgram(x, win_length, hop_size)
    M = get_mel_filterbank(S.shape[0], win_length, sr, min_freq, max_freq, n_bins)
    mel_specgram = np.log10(M.dot(S**2))
    return mel_specgram