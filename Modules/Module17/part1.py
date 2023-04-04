halfsine = lambda W: np.sin(np.pi*np.arange(W)/float(W))

def stft(x, w, h, win_fn=halfsine):
    """
    Compute the complex Short-Time Fourier Transform (STFT)
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
    ndarray(w, nwindows, dtype=np.complex) STFT
    """
    N = len(x)
    nwin = int(np.ceil((N-w)/h))+1
    # Make a 2D array
    # The rows correspond to frequency bins
    # The columns correspond to windows moved forward in time
    S = np.zeros((w, nwin), dtype=np.complex)
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
        S[:, j] = np.fft.fft(xj)
    return S

def istft(S, w, h):
    """
    Compute the complex inverse Short-Time Fourier Transform (STFT)
    Parameters
    ----------
    S: ndarray(w, nwindows, dtype=np.complex)
        Complex spectrogram
    w: int
        Window length
    h: int
        Hop length
    
    Returns
    -------
    y: ndarray(N)
        Audio samples of the inverted STFT
    """
    N = (S.shape[1]-1)*h + w # Number of samples in result
    y = np.zeros(N)
    for j in range(S.shape[1]):
        xj = np.fft.ifft(S[:, j])
        y[j*h:j*h+w] += np.real(xj)
    y /= (w/h/2)
    return y


def get_chroma_filterbank(sr, win, o1=-4, o2=4):
    """
    Compute a chroma matrix
    
    Parameters
    ----------
    sr: int
        Sample rate
    win: int
        STFT Window length
    o1: int
        Octave to start
    o2: int
        Octave to end
    
    Returns
    -------
    ndarray(12, floor(win/2)+1)
        A matrix, where each row has a bunch of Gaussian blobs
        around the center frequency of the corresponding note over
        all of its octaves
    """
    K = win//2+1 # Number of non-redundant frequency bins
    C = np.zeros((12, K)) # Create the matrix
    freqs = sr*np.arange(K)/win # Compute the frequencies at each spectrogram bin
    for p in range(12):
        for octave in range(o1, o2+1):
            fc = 440*2**(p/12 + octave)
            sigma = 0.02*fc
            bump = np.exp(-(freqs-fc)**2/(2*sigma**2))
            C[p, :] += bump
    return C

def invert_chroma(C, CS, win, hop):
    np.random.seed(0)
    SInv = (C.T).dot(CS)
    S2 = np.zeros((win, CS.shape[1]))
    S2[0:win//2+1, :] = (C.T).dot(CS)
    S2 = np.array(S2, dtype=complex)
    S2 *= np.exp(1j*2*np.pi*np.random.rand(S2.shape[0], S2.shape[1]))
    S3 = np.zeros_like(S2)
    S3[0:-1, :] = np.conj(S2[1::, :])
    S2 = S2 + S3[::-1, :]
    y = istft(S2, win, hop)
    return y
    
import numpy as np

def make_shepard(n_win, hold):
    """
    Parameters
    ----------
    n_win: int
        Number of windows to make
    hold: int
        How long to hold each note

    Returns
    -------
    ndarray(N)
        Shepard tone
    """
    win = 1024
    hop = 256
    sr = 8000
    C = get_chroma_filterbank(sr, win)
    CS = np.zeros((12, n_win))
    
    #print(C)
    ## TODO: Fill this in
    # method to create a chroma spectrogram CS which has 12 rows and n_win windows,
    # and which represents the notes 
    # 0, 1, 2, ..., 10, 11, 0, 1, 2, ..., 10, 11, ... in sequence, 
    # holding each note for hold windows, until the end is reached
    # (hint: use the modulo operator %)
    for i in range(n_win):
        CS[:,i] = C[:,i%12]
    #
    p = 0
    for i in range(12): # this goes into each row
        for j in range(n_win): # this goes into each column
            for k in range(hold): # this holds each note for hold windows
                CS[i, j] = p
                print(p)
            p = 0 if p == 11 else p + 1
        #print(CS[i, :])
        p = 0


    print(CS)

    y = invert_chroma(C, CS, win, hop)
    y = y/np.max(np.abs(y))
    return y, sr

y, sr = make_shepard(400, 10)