import numpy as np
def make_idft_signal():
    sr = 8000
    # Add a cosine that goes through 200 cycles/interval with
    # amplitude 1, and a sine that goes through 400 cycles/interval
    # with an amplitude of 2
    N = sr//2 # Interval is 1 second long
    x = np.zeros(N)
    ## TODO: Fill this in
    # Add the cosine
    n = np.arange(N)
    x += np.cos(2*np.pi*((200*n)/N))
    x += np.sin(2*np.pi*((400*n)/N))

    return x, sr