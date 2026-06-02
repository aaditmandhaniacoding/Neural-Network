# Neuron Layering
import numpy as np

input = [[1, 2, 3, 2.5],
         [2., 5., -1., 2],
         [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1],
           [.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2,3,0.5]

weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

input_array = np.array(input)
weights_array = np.array(weights)
biases_array = np.array(biases)
weights2_array = np.array(weights2)
biases2_array = np.array(biases2)

layer1_outputs = np.dot(input_array, weights_array.T) + biases_array

layer2_outputs = np.dot(layer1_outputs,weights2_array.T) + biases2_array


print(layer2_outputs)