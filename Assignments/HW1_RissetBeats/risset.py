import numpy as np
import matplotlib.pyplot as plt

def get_note_freq(p):
    """
    Return the frequency corresponding to a particular
    note number
    Parameters
    ----------
    p: int
        Note number, in halfsteps.  0 is a concert a
    """
    return 440*2**(p/12)

def load_tune(filename, tune_length):
    """
    Load in information about notes and their
    onset times from a text file
    Parameters
    ----------
    filename: string
        Path to file with the tune
    tune_length: float
        Length, in seconds, of the tune
    
    Returns
    -------
    ps: ndarray(N)
        A list of N note numbers
    times: ndarray(N)
        Duration of each note, in increments
        of sixteenth notes
    """
    tune = np.loadtxt(filename)
    ps = tune[:, 0]
    times = np.zeros(tune.shape[0])
    times[1::] = np.cumsum(tune[0:-1, 1])
    times = times*tune_length/np.sum(tune[:, 1])
    return ps, times

def do_risset_slow(filename, tune_length, freqs_per_note, sr):
    """
    Implement the naive version of Risset beats where 
    freqs_per_note sinusoids are added for every note
    Parameters
    ----------
    filename: string
        Path to file with the tune
    tune_length: float
        Length, in seconds, of the tune
    freqs_per_note: int
        Number of frequencies to use for each note
    sr: int
        The sample rate of the entire piece
    """
    ps, times = load_tune(filename, tune_length)
    ts = np.arange(int(tune_length*sr))/sr
    y = np.zeros_like(ts)
    ## TODO: Fill this in
    for i in range(len(ps)):
        for j in range(freqs_per_note):
            y += np.sin(2*np.pi*get_note_freq(ps[i])*(j+1)*ts)
    return y

def do_risset_fast(filename, tune_length, freqs_per_note, sr):
    """
    Implement the faster version of Risset beats that aggregates
    duplicate frequencies into a sine and cosine term
    Parameters
    ----------
    filename: string
        Path to file with the tune
    tune_length: float
        Length, in seconds, of the tune
    freqs_per_note: int
        Number of frequencies to use for each note
    sr: int
        The sample rate of the entire piece
    """
    ps, times = load_tune(filename, tune_length)
    ts = np.arange(int(tune_length*sr))/sr
    y = np.zeros_like(ts)
    ## TODO: Fill this in
    
    return y


