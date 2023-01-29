import numpy as np
import matplotlib.pyplot as plt
sr = 8000
t = np.linspace(0, 1, sr) # t \in [0, 1]
f = 220*t + 110*t**4
y = np.cos(2*np.pi*f)