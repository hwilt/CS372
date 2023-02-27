import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
from scipy.ndimage import maximum_filter, maximum_filter1d
from scipy.interpolate import interp2d
from instruments import *


def comb_tune(tune_filename, voice_filename, sixteenth_len, num_pulses):
    """
    Make a tune using a comb filter

    Parameters
    ----------
    tune_filename: string
        Path to a tune
    voice_filename: string
        Path to a voice audio file
    sixteenth_len: float
        Length of a sixteenth note in seconds
    num_pulses: int
        The number of pulses to use in the comb filter
    
    Returns
    -------
    y: ndarray(N)
        Audio samples from applying the comb filter
    """
    tune = np.loadtxt(tune_filename)
    notes = tune[:, 0]
    durations = tune[:, 1]*sixteenth_len
    sr, x = wavfile.read(voice_filename)
    x = np.array(x, dtype=float)/32768
    y = np.zeros_like(x) # Output audio
    idx = 0
    for note, d in zip(notes, durations):
        N = int(d*sr)
        # Pull out a chunk of audio aligned with this note
        xi = x[idx:idx+N] 
        ## TODO: Convolve this chunk with a comb 
        ## with the appropriate number of samples and
        ## with the appropriate spacing based on note
        ## and place the chunk into the final audio y
        idx += N
    return y, sr


def hann_window(N):
    """
    Create the Hann window 0.5*(1-cos(2pi*n/N))
    
    Parameters
    ----------
    N: int
        Length of window
    
    Returns
    -------
    ndarray(N): Samples of the window
    """
    return 0.5*(1 - np.cos(2*np.pi*np.arange(N)/N))

def sin_window(N):
    """
    Create the sine window sin(pi*n/N)
    
    Parameters
    ----------
    N: int
        Length of window
    
    Returns
    -------
    ndarray(N): Samples of the window
    """
    return np.sin(np.pi*np.arange(N)/N)


def blackman_harris_window(N):
    """
    Create a Blackman-Harris Window
    
    Parameters
    ----------
    N: int
        Length of window
    
    Returns
    -------
    ndarray(N): Samples of the window
    """
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168
    t = np.arange(N)/N
    return a0 - a1*np.cos(2*np.pi*t) + a2*np.cos(4*np.pi*t) - a3*np.cos(6*np.pi*t)

def stft(x, w, h, win_fn):
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

def istft(S, w, h, win_fn):
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
    win_fn: int -> ndarray(N)
        Window function (This can actually be ignored in this method)
    
    Returns
    -------
    y: ndarray(N)
        Audio samples of the inverted STFT
    """
    N = (S.shape[1]-1)*h + w # Number of samples in result
    y = np.zeros(N)
    ## TODO: Fill in y by looping through each STFT window and performing SOLA
    ## NOTE: The jth window can be accessed as S[:, j], 
    ##       and there are S.shape[1] total windows
    return y

def specgram_vocoder(tune, voice, sr, w, h, win_fn):
    """
    Perform a spectrogram vocoder

    Parameters
    ----------
    tune: ndarray(N1)
        Audio samples for the instrument sound
    voice: ndarray(N2)
        Audio samples for the voice
    sr: int
        Sample rate
    w: int
        Window length
    h: int
        Hop length
    win_fn: int -> ndarray(N)
        Window function
    
    Return
    ------
    y: ndarray(min(N1, N2))
        Audio resulting from the voice shaping the instrument sound
    """
    M = min(len(tune), len(voice))
    tune = tune[0:M]
    voice = voice[0:M]
    ## TODO: Perform the STFT of voice and the STFT of tune
    ## Then, multiply the tune STFT by the voice amplitude
    ## and invert the result into a variable "y" which will
    ## hold the final audio samples
    
    # Apply rudimentary loudness compression
    amp = maximum_filter1d(np.abs(y), int(sr))
    return y/amp


def griffin_lim(SAbs, w, h, win_fn, n_iters):
    """
    Perform Griffin-Lim Phase Retrieval on an absolute value STFT

    Parameters
    ----------
    SAbs: ndarray(w, n_win)
        Absolute magnitude spectrogram to invert
    w: int
        Window length
    h: int
        Hop length
    win_fn: int -> ndarray(N)
        Window function
    n_inters: int
        Number of iterations of phase retrieval to perform
    
    Returns
    -------
    x: ndarray(N)
        The audio retrieved from Griffin Lim
    """
    S = SAbs
    
    ## TODO: Apply n_iters iterations of the Griffin-Lin algorithm

    x = istft(S, w, h, win_fn)
    return np.real(x)

def time_shift(x, fac, w, h, win_fn, n_iters):
    """ 
    Time shift audio without changing its pitch
    by taking its absolute value STFT, stretching it 
    uniformly along the x axis, and then performing 
    Griffin-Lim phase retrieval

    Parameters
    ----------
    x: ndarray(N)
        Original audio samples
    fac: float
        Factor by which to stretch/compress audio in time
    w: int
        Window length
    h: int
        Hop length
    win_fn: int -> ndarray(N)
        Window function
    n_inters: int
        Number of iterations of phase retrieval to perform
    
    Return
    ------
    y: ndarray(N)
        Time-scaled audio samples
    """
    S = np.abs(stft(x, w, h, win_fn))
    freqs = np.arange(w)
    times = np.arange(S.shape[1])
    f = interp2d(times, freqs, S, kind = 'linear')
    new_times = np.linspace(0, times[-1], int(fac*len(times)))
    SInterp = f(new_times, freqs)
    x = griffin_lim(SInterp, w, h, win_fn, n_iters)
    return x

def im2sound(impath, w, h, win_fn, n_iters):
    """
    Turn an image into a sound by converting it into
    a complex spectrogram and inverting that spectrogram
    using Griffin-Lim

    Parameters
    ----------
    impath: string
        Path to image file
    w: int
        Window length
    h: int
        Hop length
    win_fn: int -> ndarray(N)
        Window function
    n_inters: int
        Number of iterations of phase retrieval
    
    Return
    ------
    y: ndarray(N)
        Audio samples corresponding to the inverted image
    """
    from skimage.color import rgb2gray
    import skimage.io
    X = np.flipud(rgb2gray(skimage.io.imread(impath)))
    nwin = X.shape[1]
    S = np.zeros((w, nwin))
    ## TODO: Fill in the spectrogram with the image X
    ## and its mirror image so that the inverse will be real
    plt.figure(figsize=(8, 8))
    plt.imshow(S, aspect='auto', cmap='magma_r')
    plt.gca().invert_yaxis()
    plt.xlabel("Window Index")
    plt.ylabel("Frequency Index")
    plt.savefig("S.png", bbox_inches='tight')
    return griffin_lim(S, w, h, win_fn, n_iters)


def make_beepy_tune(x, w, h, win_fn, time_win, freq_win, max_freq, n_iters):
    """ 
    Make a spectrogram consisting only of 1s where the maxes of
    the original spectrogram are, and 0 everywhere else.  Then
    invert the new spectrogram with Griffin-Lim to obtain a beepy
    tune that tricks Shazam

    Parameters
    ----------
    x: ndarray(N)
        Original audio samples
    w: int
        Window length
    h: int
        Hop length
    win_fn: int -> ndarray(N)
        Window function
    time_win: int
        Time window half-length
    freq_win: int
        Frequency window half-length
    max_freq: int
        Only consider maxes up to this frequency index
    n_inters: int
        Number of iterations of phase retrieval to perform
    
    Return
    ------
    y: ndarray(N)
        Beepy tune
    """
    pass
    ## TODO: Fill this in

def pitch_shift(x, shift, w, h, win_fn, n_iters):
    """
    Pitch shift audio without changing the timing by taking 
    its absolute value STFT and warping it nonlinearly,
    then performing Griffin-Lim phase retrieval

    Parameters
    ----------
    x: ndarray(N)
        Original audio samples
    shift: int
        Number of halfsteps by which to shift audio
    w: int
        Window length
    h: int
        Hop length
    win_fn: int -> ndarray(N)
        Window function
    n_inters: int
        Number of iterations of phase retrieval to perform
    
    Return
    ------
    y: ndarray(N)
        Pitch shifted audio samples
    """
    S = np.abs(stft(x, w, h, win_fn))
    ## TODO: Fill this in to create a spectrogram S2
    ## which is a frequency warped version of S, and invert this
    ## with griffin lim to get the pitch shifted audio

    N = S.shape[0]
    plt.figure(figsize=(18, 6))
    plt.subplot(121)
    plt.imshow(np.log10(np.abs(S)), aspect='auto', cmap='magma_r')
    plt.gca().invert_yaxis()
    plt.xlabel("Time Index")
    plt.ylabel("Frequency Index")
    plt.title("Original Spectrogram")
    plt.subplot(122)
    plt.imshow(np.log10(np.abs(S2)), aspect='auto', cmap='magma_r', interpolation='nearest')
    plt.gca().invert_yaxis()
    plt.xlabel("Time Index")
    plt.ylabel("Frequency Index")
    plt.title("Spectrogram Shifted by {} Halfsteps".format(shift))
    plt.savefig("PitchShift.svg", bbox_inches='tight')
    return x