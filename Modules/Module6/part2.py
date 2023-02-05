import numpy as np
def make_lowAString():
    sr = 22050
    t = np.arange(int(sr/2))/sr
    fc = 110
    fm = 110
    # TODO: Make this decay as 5e^(-8t)
    I = 5 * np.exp(-8*t)
    # TODO: Do FM synthesis here
    y = np.cos(2*np.pi*fc*t + I*np.sin(2*np.pi*fm*t))
    return y, sr