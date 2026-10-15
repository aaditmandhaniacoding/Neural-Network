# Neuron Layering
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

learning_rate = .001


class Activation_Relu:
    def forward(self, inputs):
        self.inputs = inputs

        # calculate output values from input
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        # Since we need to modify the original variable
        # lets make a copy of values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

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

class Activation_Softmax_Loss_CategoricalCrossentropy:
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_SoftMax()
        self.loss = Loss_CategoricalentropyLoss()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
    
# Dense Layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # creates a matrix of random numbers drawn from a normal (Gaussian) distribution with mean: 0 and SD: 1
        
        self.biases = np.zeros((1,n_neurons)) # creates a row with n_neurons amount of columns of zeroes

    def forward(self, inputs):
        self.inputs = inputs
        # calculate output values from input, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients of parameter
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)
        # Gradients on values
        self.dinputs = np.dot(dvalues, self.weights.T)
# creates a dataset
X, y = spiral_data(samples = 100, classes = 3)
# creates a dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2,3)
# create Relu activation (will be used with dense layer):
activation1 = Activation_Relu()
# create a second Dense layer with 3 input features (take output from previous layer here) and three output values
dense2 = Layer_Dense(3, 3)
# create a Softmax activation (will be used with dense layer):
activation2 = Activation_SoftMax()
# create a loss function
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()


# perform a forward pass of the training data through this layer
dense1.forward(X)
#Make a forward pass through activation func
# output of first layer in this
activation1.forward(dense1.output)
# Make a forward pass through second Dense layer
# output of second layer in this
dense2.forward(activation1.output)
# perform a forward pass through the activation/loss function
# take the output of second dense layer here and returns loss 
loss = loss_activation.forward(dense2.output, y)
# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis = 1)
if len(y.shape) == 2:
    y = np.argmax(y, axis = 1)
accuracy = np.mean(predictions == y)


# backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)

print(loss_activation.output[:5])
print("loss:", loss)
print("accuracy:", accuracy)
