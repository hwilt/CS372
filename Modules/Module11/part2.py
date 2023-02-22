import numpy as np
def get_blackman_harris(N):
    """
    Compute the Blackman-Harris Window for a window N samples long
    """
    w = np.zeros(N)
    ## TODO: Fill this in
    # w[n] = 0.42 - 0.5 \cos(2 \pi n / N) + 0.08 \cos(4 \pi n / N)
    for n in range(N):
        w[n] = 0.42 - 0.5 * np.cos(2 * np.pi * n / N) + 0.08 * np.cos(4 * np.pi * n / N)
    return w
    