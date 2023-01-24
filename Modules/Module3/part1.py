import numpy as np
sr = 44100
f0 = 220
t = np.arange(sr*2)/sr
decay = np.exp(-2*t)
y = np.zeros_like(t)
y += np.cos(2*np.pi*f0*t)*decay # Fundamental frequency
# TODO: Add a loop which adds on the first 19 harmonics after the base frequency
for x in range(2,21):
    y += np.cos(2*np.pi*x*f0*t)*(decay**x)


