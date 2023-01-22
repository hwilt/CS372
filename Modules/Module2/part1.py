import numpy as np
import matplotlib.pyplot as plt
# TODO: Change x to be 90 samples in [-3, 3]
#       and y to be f(x) at those samples
#      (f(x) = x^3 - 10x)
x = np.linspace(-3, 3, 90)
y = np.array([x**3 - 10*x for x in x])

# Use matplotlib's histogram function to plot a histogram
plt.figure(figsize=(6, 4), dpi=100)
plt.plot(x, y)
plt.title("$f(x) = x^3 - 10x$")
plt.xlabel("x")
plt.ylabel("y")
#save_figure_js()
plt.show()