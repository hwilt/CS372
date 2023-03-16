import numpy as np

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


def audio_onset_fn(x, w=2048, h=512):
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
    diff[diff < 0] = 0
    novfn = np.sum(diff, axis=0)
    return novfn

def autocorr(x):
    """
    Perform FFT-based autocorrelation of a signal

    Parameters
    ----------
    x: ndarray(N)
        The signal to process
    
    Returns
    -------
    r: ndarray(N)
        The autocorrelation function
    """
    N = len(x)
    xpad = np.zeros(N*2)
    xpad[0:N] = x
    F = np.fft.fft(xpad)
    FConv = np.real(F)**2 + np.imag(F)**2 # Fourier transform of the convolution of x and its reverse
    return np.real(np.fft.ifft(FConv)[0:N])


def estimate_tempo(novfn, hop_length, sr):
    """
    Parameters
    ----------
    novfn: ndarray(N)
        An audio novelty function
    hop_length: int
        Hop between samples in the audio novelty function
    sr: int
        The sample rate
    
    Returns
    -------
    float: Estimated tempo, in beats per minute
    """
    novfncopy = np.array(novfn) # Make a copy of the novelty function
    novfncopy[0:2] = 0 # Ignore the first few bins
    tempo = 0
    ## TODO: Fill this in.  Use np.argmax(novfncopy) to find
    ## the index of the novelty function corresponding to the maximum
    ## peak, then convert this index into beats per minute using
    ## the hop length and sample rate
    tempo = 60*sr/hop_length/np.argmax(novfncopy)
    return tempo