import numpy as np

# linear function
def linear(x):
    return x

# binary function
def binary(x):
    if x < 0:
        return 0
    else:
        return 1

# sigmoid function
def sigmoid(x):
    return (1/(1+np.exp(-x)))

# relu function
def relu(x):
    if x < 0:
        return 0
    else:
        return x

# softmax function
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

# tanh function
def tanh(x):
    return np.tanh(x)