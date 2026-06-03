# Neuron Layering
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
# Dense Layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # creates a matrix of random numbers drawn from a normal (Gaussian) distribution with mean: 0 and SD: 1
        
        self.biases = np.zeros((1,n_neurons)) # creates a row with n_neurons amount of columns of zeroes

    def forward(self, inputs):
        # calculate output values from input, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases
# creates a dataset
X, y = spiral_data(samples = 100, classes = 3)
# creates a dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2,3)
# perform a forward pass of the training data through this layer
dense1.forward(X)

print(dense1.output[:5])

class Activation_Relu:
    def forward(self, inputs):
        # calculate output values from input
        self.output = np.maximum(0, inputs)
class Activation_SoftMax:
    pass