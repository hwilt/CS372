import numpy as np
def make_echo():
    x, sr = get_gameover()
    h = np.zeros(2000)
    ## TODO: Make every 200th sample in h a 1
    ## and convolve x with h, saving the result to y
    for i in range(0,len(h), 200):
        h[i] = 1
    y = np.convolve(x, h)
    return y, sr