import numpy as np
def Hebbian_update(alpha, neurons):
    new_weights=np.matmul(neurons, neurons.T)
    new_weights=new_weights*alpha
    np.fill_diagonal(new_weights, 0)
    return new_weights