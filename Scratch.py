# Neuron Layering
import numpy as np

# Dense Layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # creates a matrix of random numbers drawn from a normal (Gaussian) distribution with mean: 0 and SD: 1
        
        self.biases = np.zeros((1,n_neurons)) # creates a row with n_neurons amount of columns of zeroes

    def forward(self, inputs):
        # calculate output values from input, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases
