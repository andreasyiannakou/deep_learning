
import matplotlib.pyplot as plt
import numpy as np


def plot_activation_function(name, formula):
    x = np.arange(-10., 10., 0.2)
    y = formula(x)
    # plot the function
    fig, ax = plt.subplots()
    plt.plot(x, y)
    ax.set_title(name)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    # delete the variables
    del x, y, fig, ax

# linear function
def linear(x):
    return x

# binary function
def binary(x):
    a = []
    for item in x:
        if item < 0:
            a.append(0)
        else:
            a.append(1)
    return a

# sigmoid function
def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+np.exp(-item)))
    return a

# relu function
def relu(x):
    a = []
    for item in x:
        if item < 0:
            a.append(0)
        else:
            a.append(item)
    return a

# relu function
def lrelu(x):
    a = []
    for item in x:
        if item < 0:
            a.append(0.1*item)
        else:
            a.append(item)
    return a

# relu function
def elu(x):
    a = []
    for item in x:
        if item < 0:
            a.append(1*(np.exp(item)-1))
        else:
            a.append(item)
    return a

# softmax function
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

# tanh function
def tanh(x):
    return np.tanh(x)

"""
plot_activation_function('Linear Function', linear)
plot_activation_function('Binary Function', binary)
plot_activation_function('Sigmoid Function', sigmoid)
plot_activation_function('Softmax Function', softmax)
plot_activation_function('Tanh Function', tanh)
# eLU functions
plot_activation_function('ReLU Function', relu)
plot_activation_function('Leaky ReLU Function', lrelu)
plot_activation_function('ELU Function', elu)
"""
