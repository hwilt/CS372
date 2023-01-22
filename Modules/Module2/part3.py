import numpy as np
import IPython.display as ipd
def get_cosine(N, sr, pitch):
    """
    Return a cosine wave at a particular pitch, assuming
    an equal tempered scale
    parameters
    ----------
    N: int
      Number of samples
    sr: int
      Sample rate
    pitch: int
      Pitch
    """
    f = 440*2**(pitch/12)
    t = np.arange(N)/sr
    return np.cos(2*np.pi*f*t)

## TODO: Change y to hold the requested samples of cosines
'''
0.25 seconds of pitch 0
0.25 seconds of silence
0.25 seconds of pitch 0
0.25 seconds of silence
1 second of pitch 2
1 second of pitch 0
1 second of pitch 5
2 seconds of pitch 4
'''
sr = 8000
y = np.array([0])
y = np.concatenate((y, get_cosine(int(sr/4), sr, 0)))
y = np.concatenate((y, np.zeros(int(sr/4))))
y = np.concatenate((y, get_cosine(int(sr/4), sr, 0)))   
y = np.concatenate((y, np.zeros(int(sr/4))))
y = np.concatenate((y, get_cosine(int(sr), sr, 2)))
y = np.concatenate((y, get_cosine(int(sr), sr, 0)))
y = np.concatenate((y, get_cosine(int(sr), sr, 5)))
y = np.concatenate((y, get_cosine(int(2*sr), sr, 4)))
y = np.sign(y)



ipd.Audio(y, rate=sr)