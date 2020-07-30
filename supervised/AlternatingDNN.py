import numpy as np
from sklearn import datasets


def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ = 0 if (Z <= 0) else dZ
    return dZ

def init_layers(nn_architecture, seed=99):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1
    return params_values

def forwardAlter(inp, W, act='relu'):
    # aW = np.ones(W.shape) - W #  Inverse Weights
    z_curr = np.dot(inp, W)
    if act == 'relu':
        actf = relu
    elif act == 'sigmoid':
        actf = sigmoid
    else:
        raise Exception("Unavailable activation function")
    return actf(z_curr), z_curr


arch = init_layers([
    {"input_dim": 3, "output_dim":4},
    {"input_dim": 4, "output_dim":4},
    {"input_dim": 4, "output_dim":3}])
print(arch)

iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target
Xtrain = X[len(X)//10:]
Ytrain = Y[len(Y)//10:]
XEval = X[:len(Y)//10]
YEval = Y[:len(Y)//10]



