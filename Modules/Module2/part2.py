import numpy as np
import IPython.display as ipd
## TODO: Change y to hold the requested samples of the square wave
sr = 44100
t = np.arange(int(sr/2))/sr
y = np.array([np.sign(np.cos(2*np.pi*660*t)) for t in t])

ipd.Audio(y, rate=sr)