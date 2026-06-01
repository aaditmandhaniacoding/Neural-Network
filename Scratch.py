# Neuron Layering
input = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1],
           [.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2,3,0.5]

bias1 = 2
bias2 = 3
bias3 = 0.5

outputs = [
    # neuron 1:
    input[0]*weights[0][0]+
    input[1]*weights[0][1]+
    input[2]*weights[0][2]+
    input[3]*weights[0][3]+ bias1,
    # neuron 2:
    input[0]*weights[1][0]+
    input[1]*weights[1][1]+
    input[2]*weights[1][2]+
    input[3]*weights[1][3]+ bias2,
    # neuron 3:
    input[0]*weights[2][0]+
    input[1]*weights[2][1]+
    input[2]*weights[2][2]+
    input[3]*weights[2][3]+ bias3
]


print(outputs)