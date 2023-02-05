import numpy as np
def make_fm(t, fc, fm, d):
    """
    t: ndarray(N)
        Time samples at which to compute the waveform
    fc: float
        Carrier frequency
    fm: float
        Modulation frequency
    d: float
        Peak frequency deviation (amplitude of fm)
    """
    ## TODO: Fill this in to generate the requested note
    ## for fm synthesis
    return np.cos(2*np.pi*fc*t + (d/fm)*np.sin(2*np.pi*fm*t))