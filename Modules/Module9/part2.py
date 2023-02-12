def index_to_freq(k, N, sr):
    """
    Given a particular frequency index in either the
    cosine or sine array of DFT dot products, as well
    as the number of samples in the signal and the sample
    rate of the signal, return the frequency associated
    to the sine or cosine at a particular index $k$ in hz

    Parameters
    ----------
    k: int
        Frequency index
    N: int
        Number of samples
    sr: int
        Sample rate
    Returns
    -------
    """
    return k * sr / N