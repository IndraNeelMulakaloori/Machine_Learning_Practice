import numpy as np 
import matplotlib.pyplot as plt 
# Defining sigmoid func
def sigmoid(z):
    """    
      Compute the sigmoid of z
    Args:
        z (ndarray): A scalar, numpy array of any size.
    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """
    z = np.clip(z, -500, 500)
    return 1.0/(1.0 + np.exp(-z))
