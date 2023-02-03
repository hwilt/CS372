import numpy as np
def make_zero_vowels(x, win, thresh):
    zcs = np.sign(x)
    zcs = 0.5*np.abs(zcs[1::] - zcs[0:-1])
    zcs_win = np.cumsum(zcs)[win::] - np.cumsum(zcs)[0:-win]
    y = x[int(win/2):int(win/2)+len(zcs_win)]
    ### TODO: Make audio with vowels zero, according
    ### to number of zero crossings in each window compared
    ### to the threshold
    y[zcs_win < thresh] = 0
    
    return y