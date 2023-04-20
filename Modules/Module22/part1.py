import numpy as np

def softmax(u):
    """
    Implement the softmax method

    Parameters
    ----------
    u: ndarray(N)
      Input to softmax
      
    Returns
    -------
    ndarray
        Result of softmax
    """
    ret = np.zeros(len(u))
    ## TODO: Fill this in
    # formula: softmax(u)[i] = e^u[i]/sum(e^u[j])
    # for i in range(len(u)):
    #     ret[i] = np.exp(u[i])/np.sum(np.exp(u))
    ret = np.exp(u)/np.sum(np.exp(u))
    
    return ret

np.random.seed(0)
u = np.random.randn(10)
res = softmax(u)