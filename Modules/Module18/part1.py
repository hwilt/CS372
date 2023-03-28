import numpy as np

halfsine = lambda W: np.sin(np.pi*np.arange(W)/float(W))

def iSTFT(pS, W, H, winfunc = halfsine):
    """
    :param pS: An NBins x NWindows spectrogram
    :param W: A window size
    :param H: A hopSize
    :param winfunc: Handle to a window function
    :returns S: Spectrogram
    """
    #First put back the entire redundant STFT
    S = np.array(pS, dtype = np.complex128)
    if W%2 == 0:
      #Even Case
      S = np.concatenate((S, np.flipud(np.conj(S[1:-1, :]))), 0)
    else:
      #Odd Case
      S = np.concatenate((S, np.flipud(np.conj(S[1::, :]))), 0)

    #Figure out how long the reconstructed signal actually is
    N = W + H*(S.shape[1] - 1)
    X = np.zeros(N, dtype = np.complex128)

    #Setup the window
    Q = W/H
    if Q - np.floor(Q) > 0:
      print('Warning: Window size is not integer multiple of hop size')
    if not winfunc:
      #Use half sine by default
      winfunc = halfsine
    win = winfunc(W)
    win = win/(Q/2.0)

    #Do shift overlap/add synthesis
    for i in range(S.shape[1]):
      X[i*H:i*H+W] += win*np.fft.ifft(S[:, i])
    return np.real(X)


def get_rhythm_loop():
    """
    Returns
    -------
    ndarray(N)
        A rhythm loop where every other beat switches the window that's being activated
    """
    win_length = 2048*3
    hop_length = 512
    sr = 22050
    SBoth = get_SBoth()
    H = np.zeros((2, 500))
    ## TODO: Fill in both rows of H
    # create a matrix H with two rows to create a rhythm where a beat happens every 20 samples, but the beat sound switches every other beat
    H[0, 0::40] = 1
    H[1, 20::40] = 1
    V = SBoth.dot(H)
    y = iSTFT(V, win_length, hop_length)

    return y, sr