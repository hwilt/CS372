import numpy as np
def get_contour(freqs):
    """
    Code adapted from
    https://www.audiolabs-erlangen.de/resources/MIR/FMP/C1/C1S3_Dynamics.html
    Parameters
    ----------
    freqs: ndarray(N)
        Frequencies at which to sample the equal intensity contour
    
    Returns
    -------
    ndarray(N): Amplitudes at corresponding frequencies that are equal in intensity
    """
    freq = 1000
    h_freq = ((1037918.48 - freq**2)**2 + 1080768.16 * freq**2) / ((9837328 - freq**2)**2 + 11723776 * freq**2)
    n_freq = (freq / (6.8966888496476 * 10**(-5))) * np.sqrt(h_freq / ((freq**2 + 79919.29) * (freq**2 + 1345600)))
    h_freq_range = ((1037918.48 - freqs**2)**2 + 1080768.16 * freqs**2) / ((9837328 - freqs**2)**2 + 11723776 * freqs**2)
    n_freq_range = (freqs / (6.8966888496476 * 10**(-5))) * np.sqrt(h_freq_range / ((freqs**2 + 79919.29) * (freqs**2 + 1345600)))
    equal_loudness_contour = np.abs(n_freq / n_freq_range)
    return equal_loudness_contour


def get_equal_intensity_chirp(t, freqs):
    """
    Construct an equal intensity chirp given
    some frequency trajectory

    Parameters
    ----------
    t: ndarray(N)
    Times (seconds)
    freqs: ndarray(N)
    Instantaneous frequency trajectory over time (hz)
    """
    f = np.cumsum(freqs)*(t[1]-t[0])
    y = 0.1*np.cos(2*np.pi*f)
    y[0] = 1 # To keep amplitude down
    ## TODO: Scale y so that it's equally intense perceptually
    ## everywhere
    y = y * get_contour(freqs)
    return y