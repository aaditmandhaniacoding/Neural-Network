# Neuron Layering
import numpy as np
import nnfs
from nnfs.datasets import vertical_data
nnfs.init()

class Activation_Relu:
    def forward(self, inputs):
        # calculate output values from input
        self.output = np.maximum(0, inputs)
class Activation_SoftMax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)

        self.output = probabilities
class Loss:
    def calculate(self, output, y):
        # calculate the sample losses
        sample_losses = self.forward(output,y)
        # mean loss
        data_loss = np.mean(sample_losses)

        return data_loss

class Loss_CategoricalentropyLoss(Loss):
    def forward(self, y_pred, y_true):
        # num samples in batch
        samples = len(y_pred)
        # clip it so that it never reaches 0 or 1
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape)==1:
            correct_confidence = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape)==2:
            correct_confidence = np.sum(y_pred_clipped*y_true, axis = 1)
        negative_log_likelihoods = -np.log(correct_confidence)
        return negative_log_likelihoods


# Dense Layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # creates a matrix of random numbers drawn from a normal (Gaussian) distribution with mean: 0 and SD: 1
        
        self.biases = np.zeros((1,n_neurons)) # creates a row with n_neurons amount of columns of zeroes

    def forward(self, inputs):
        # calculate output values from input, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases
# creates a dataset
X, y = vertical_data(samples = 100, classes = 3)
# creates a dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2,3)
# create Relu activation (will be used with dense layer):
activation1 = Activation_Relu()
# create a second Dense layer with 3 input features (take output from previous layer here) and three output values
dense2 = Layer_Dense(3, 3)
# create a Softmax activation (will be used with dense layer):
activation2 = Activation_SoftMax()
# create a loss function
loss_function = Loss_CategoricalentropyLoss()

#Helper Variables
lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(10000):
    dense1.weights += .05 * np.random.randn(2, 3)
    dense1.biases += .05 * np.random.randn(1, 3)
    dense2.weights += .05 * np.random.randn(3, 3)
    dense2.biases += .05 * np.random.randn(1, 3)
    # perform a forward pass of the training data through this layer
    dense1.forward(X)

    #Make a forward pass through activation func
    # output of first layer in this
    activation1.forward(dense1.output)
    #Make a forward pass through second Dense layer
    # output of second layer in this
    dense2.forward(activation1.output)
    #Make a forward pass through activation func
    # output of second layer in this
    activation2.forward(dense2.output)
    loss = loss_function.calculate(activation2.output, y)

    #Calculate accuracy from output of activation2 and targets
    #calculate values along first axis
    predictions = np.argmax(activation2.output, axis = 1)
    accuracy = np.mean(predictions == y)

    if loss < lowest_loss:
        print("New set of weights found, iteration", iteration, "loss:", loss, "acc:", accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()