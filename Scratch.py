# Neuron Layering
import numpy as np

input = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1],
           [.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2,3,0.5]

bias1 = 2
bias2 = 3
bias3 = 0.5

layer_output = []
# Using For Loops to create a simpler and scalable network
layer_output = np.dot(weights,input)+biases

print(layer_output)