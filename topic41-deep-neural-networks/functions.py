import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_activation(fn):
    z = np.arange(-10, 10, 0.2)
    y = fn(z)
    dy = fn(z, derivative=True)
    fig,ax=plt.subplots(figsize=(6,4))
    ax.set_title(f'{fn.__name__}')
    ax.set(xlabel='Input',ylabel='Output')
    ax.axhline(color='gray', linewidth=1,)
    ax.axvline(color='gray', linewidth=1,)
    ax.plot(z, y, 'r', label='original (y)')
    ax.plot(z, dy, 'b', label='derivative (dy)')
    ax.legend();
    plt.show()
    
    
def sigmoid(x, derivative=False):
    f = 1 / (1 + np.exp(-x))
    if (derivative == True):
        return f * (1 - f)
    return f

def tanh(x, derivative=False):
    f = np.tanh(x)
    if (derivative == True):
        return (1 - (f ** 2))
    return np.tanh(x)

def relu(x, derivative=False):
    f = np.zeros(len(x))
    if (derivative == True):
        for i in range(0, len(x)):
            if x[i] > 0:
                f[i] = 1  
            else:
                f[i] = 0
        return f
    for i in range(0, len(x)):
        if x[i] > 0:
            f[i] = x[i]  
        else:
            f[i] = 0
    return f

def leaky_relu(x, leakage = 0.05, derivative=False):
    f = np.zeros(len(x))
    if (derivative == True):
        for i in range(0, len(x)):
            if x[i] > 0:
                f[i] = 1  
            else:
                f[i] = leakage
        return f
    for i in range(0, len(x)):
        if x[i] > 0:
            f[i] = x[i]  
        else:
            f[i] = x[i]* leakage
    return f

def arctan(x, derivative=False):
    if (derivative == True):
        return 1/(1+np.square(x))
    return np.arctan(x)
