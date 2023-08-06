# %% [markdown]
# # Multiple Linear Regression 
# Multiple linear regression is a statistical method that uses multiple independent variables to predict a single dependent variable. <br>
# The goal of multiple linear regression is to find a linear relationship between the dependent variable and the independent variables.
# 
# The Model would be : 
# $$
#         f_w,_b(x^{->}) = w^{->} . x^{->} + b
# $$
# <em>This above Notation is  </em> **Vector Notation**

# %% [markdown]
# # 2 Problem Statement 
# | Size (sqft) | Number of Bedrooms  | Number of floors | Age of  Home | Price (1000s dollars)  |   
# | ----------------| ------------------- |----------------- |--------------|-------------- |  
# | 2104            | 5                   | 1                | 45           | 460           |  
# | 1416            | 3                   | 2                | 40           | 232           |  
# | 852             | 2                   | 1                | 35           | 178           |  
# 

# %%
# Train a model 
import numpy as np 
import matplotlib as mpl 



# %%
# Single Prediction , Vector 
def predict(x, w, b): 
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b     
    return p    

# %%

def compute_cost(X,y,w,b):
    m = X.shape[0]
    n = X.shape[1]

    cost = 0.0
    
    for i in range(m):
        f_w_b_i = np.dot(X[i],w) + b
        cost += (f_w_b_i - y[i]) ** 2
    return cost/(2*m) 




# %%


# %%
def compute_gradient(X,y,w,b):
    m = X.shape[0]
    n = X.shape[1]
    dj_dw = np.zeros(n)
    dj_db = 0.
    for i in range(m):
        err = (np.dot(X[i],w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i, j]
        dj_db += err
    dj_dw /= m
    dj_db /= m 
    return dj_db,dj_dw

# %%
#Compute and display gradient 


# %%
import copy
import math
def gradient_descent(X, y, w_in, b_in,alpha, num_iters):
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
    # number of training examples
    m = len(X)
    
   # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = compute_gradient(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( compute_cost(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters/10) == 0:
             print(f"{i:9d} {J_history[-1]:0.5e} {w[0]: 0.1e} {w[1]: 0.1e} {w[2]: 0.1e} {w[3]: 0.1e} {b: 0.1e} {dj_dw[0]: 0.1e} {dj_dw[1]: 0.1e} {dj_dw[2]: 0.1e} {dj_dw[3]: 0.1e} {dj_db: 0.1e}")
        
    return w, b, J_history #return final w,b and J history for graphing

        

    




# %%
