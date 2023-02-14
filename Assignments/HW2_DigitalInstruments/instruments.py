import numpy as np
import matplotlib.pyplot as plt


def karplus_strong_note(sr, note, duration, decay):
    """
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    decay: float 
        Decay amount (between 0 and 1)

    Returns
    -------
    ndarray(N): Audio samples for this note
    """
    N = int(duration*sr)
    y = np.zeros(N)
    ## TODO: Fill this in
    T = int(sr/(440*2**(note/12)))
    y[:T] = np.random.randn(T)
    for i in range(T, N):
        y[i] = decay * (y[i - T] + y[i - T + 1])/2
    return y

def fm_synth_note(sr, note, duration, ratio=2, I=2, 
                  envelope = lambda N, sr: np.ones(N),
                  amplitude = lambda N, sr: np.ones(N)):
    """
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    ratio: float
        Ratio of modulation frequency to carrier frequency
    I: float
        Modulation index (ratio of peak frequency deviation to
        modulation frequency)
    envelope: function (N, sr) -> ndarray(N)
        A function for generating an ADSR profile
    amplitude: function (N, sr) -> ndarray(N)
        A function for generating a time-varying amplitude

    Returns
    -------
    ndarray(N): Audio samples for this note
    """
    N = int(duration*sr)
    y = np.zeros(N)
    ## TODO: Fill this in
    # y(t) = A(t)cos(2pi*fc*t + I(t)sin(2pi*fm*t))
    fc = 440*2**(note/12)
    fm = ratio*fc
    t = np.arange(N)/sr
    y = amplitude(N, sr)*np.cos(2*np.pi*fc*t + (envelope(N,sr)*I)*np.sin(2*np.pi*fm*t))
    return y

def exp_env(N, sr, mu=3):
    """
    Make an exponential envelope
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate
    mu: float
        Exponential decay rate: e^{-mu*t}

    Returns
    -------
    ndarray(N): Envelope samples
    """
    return np.exp(-mu*np.arange(N)/sr)

def drum_like_env(N, sr):
    """
    Make a drum-like envelope, according to Chowning's paper
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate

    Returns
    -------
    ndarray(N): Envelope samples
    """
    ## TODO: Fill this in
    return np.zeros(N)

def wood_drum_env(N, sr):
    """
    Make the wood-drum envelope from Chowning's paper
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate

    Returns
    -------
    ndarray(N): Envelope samples
    """
    ## TODO: Fill this in
    return np.zeros(N)

def brass_env(N, sr):
    """
    Make the brass ADSR envelope from Chowning's paper
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate
    
    Returns
    -------
    ndarray(N): Envelope samples
    """
    ## TODO: Fill this in
    # Attack = 0.1s
    # Decay = 0.1s
    # Sustain = All of the envelope up to the release, which is a very gradual decay
    # Release = the last 0.1s
    # if the note is shorter than 0.3 seconds, completely cut out the sustain, then, the as much of the release as you need.  If it's shorter than 0.2 seconds, then the release is also completely gone, and start cutting the decay.  If it's shorter than 0.1 seconds, then cut the attack.
    total_seconds = N/sr
    attack = np.zeros(N)
    decay = np.zeros(N)
    sustain = np.zeros(N)
    release = np.zeros(N)
    if total_seconds < 0.1:
        attack = np.linspace(0, N, 3)
    else:
        attack = np.linspace(0, (0.1*sr), 3)
    if total_seconds < 0.2:
        decay = np.linspace((0.1*sr), N, 4)
    else:
        decay = np.linspace((0.1*sr), (0.2*sr), 4)
    if total_seconds < 0.3:
        release = np.linspace((0.2*sr), N, 4)
    else:
        sustain = np.linspace((0.2*sr), (total_seconds-0.1)*sr, 4)
        release = linspace((0.2*sr), N, 4)
    return np.concatenate((attack, decay, sustain, release))


def dirty_bass_env(N, sr):
    """
    Make the "dirty bass" envelope from Attack Magazine
    https://www.attackmagazine.com/technique/tutorials/dirty-fm-bass/
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate
    
    Returns
    -------
    ndarray(N): Envelope samples
    """
    ## TODO: Fill this in
    return np.zeros(N)

def fm_plucked_string_note(sr, note, duration, mu=3):
    """
    Make a plucked string of a particular length
    using FM synthesis
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    mu: float
        The decay rate of the note
    
    Returns
    -------
    ndarray(N): Audio samples for this note
    """
    envelope = lambda N, sr: exp_env(N, sr, mu)
    return fm_synth_note(sr, note, duration,
                ratio = 1, I = 8, envelope = envelope,
                amplitude = envelope)

def fm_electric_guitar_note(sr, note, duration, mu=3):
    """
    Make an electric guitar string of a particular length by
    passing along the parameters to fm_plucked_string note
    and then turning the samples into a square wave

    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    mu: float
        The decay rate of the note
    
    Return
    ------
    ndarray(N): Audio samples for this note
    """
    return np.sign(fm_plucked_string_note(sr, note, duration, mu))

def fm_brass_note(sr, note, duration):
    """
    Make a brass note of a particular length
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    
    Return
    ------
    ndarray(N): Audio samples for this note
    """
    ## TODO: Fill this in
    envelope = lambda N, sr: brass_env(N, sr)
    return fm_synth_note(sr, note, duration,
                ratio = 1, I = 10, envelope = envelope,
                amplitude = envelope)


def fm_bell_note(sr, note, duration):
    """
    Make a bell note of a particular length
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    
    Returns
    -------
    ndarray(N): Audio samples for this note
    """
    ## TODO: Fill this in
    envelope = lambda N, sr: exp_env(N, sr, 0.8)
    return fm_synth_note(sr, note, duration,
                ratio = 1.4, I = 2, envelope = envelope,
                amplitude = envelope)


def fm_drum_sound(sr, note, duration, fixed_note = -14):
    """
    Make what Chowning calls a "drum-like sound"
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number (which is ignored)
    duration: float
        Seconds of audio
    fixed_note: int
        Note number of the fixed note for this drum
    
    Returns
    ------
    ndarray(N): Audio samples for this drum hit
    """
    ## TODO: Fill this in
    return None # This is a dummy value

def fm_wood_drum_sound(sr, note, duration, fixed_note=-14):
    """
    Make what Chowning calls a "wood drum sound"
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number (which is ignored)
    duration: float
        Seconds of audio
    fixed_note: int
        Note number of the fixed note for this drum
    
    Returns
    -------
    ndarray(N): Audio samples for this drum hit
    """
    ## TODO: Fill this in
    return None # This is a dummy value

def snare_drum_sound(sr, note, duration):
    """
    Make a snare drum sound by shaping noise
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number (which is ignored)
    duration: float
        Seconds of audio
    fixed_note: int
        Note number of the fixed note for this drum
    
    Returns
    -------
    ndarray(N): Audio samples for this drum hit
    """
    ## TODO: Fill this in
    return None # This is a dummy value

def fm_dirty_bass_note(sr, note, duration):
    """
    Make a "dirty bass" note, based on 
    https://www.attackmagazine.com/technique/tutorials/dirty-fm-bass/
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number (which is ignored)
    duration: float
        Seconds of audio
    
    Returns
    -------
    ndarray(N): Audio samples for this drum hit
    """
    ## TODO: Fill this in
    return None # This is a dummy value

def make_tune(filename, sixteenth_len, sr, note_fn):
    """
    Parameters
    ----------
    filename: string
        Path to file containing the tune.  Consists of
        rows of <note number> <note duration>, where
        the note number 0 is a 440hz concert A, and the
        note duration is in factors of 16th notes
    sixteenth_len: float
        Length of a sixteenth note, in seconds
    sr: int
        Sample rate
    note_fn: function (sr, note, duration) -> ndarray(M)
        A function that generates audio samples for a particular
        note at a given sample rate and duration
    
    Returns
    -------
    ndarray(N): Audio containing the tune
    """
    tune = np.loadtxt(filename)
    notes = tune[:, 0]
    durations = sixteenth_len*tune[:, 1]
    ## TODO: Fill this in
    y = np.array([])
    for note, duration in zip(notes, durations):
        if np.isnan(note):
            y = np.concatenate((y, np.zeros(int(duration*sr))))
        y = np.concatenate((y, note_fn(sr, note, duration)))
    return y