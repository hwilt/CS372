import numpy as np

logistic = lambda u: 1/(1+np.exp(-u))

def nn_forward(Ws, bs, x):
    """
    Evaluate a feedforward neural network

    Parameters
    ----------
    Ws: list of ndarray(N_{i+1}, Ni)
        Weight matrices for each layer
    bs: list of ndarray(N_{i+1})
        Bias vectors for each layer
    x: ndarray(N_0)
        Input
      
    Returns
    -------
    ndarray
        Result of neural network evaluation
    """
    ## TODO: Evaluate the neural network on an input x
    for i in range(len(Ws)):
        x = logistic(np.dot(Ws[i], x) + bs[i])
    return x