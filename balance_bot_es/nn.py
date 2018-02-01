import numpy as np

class NeuralNet():

    def __init__(self, sizes):
        self.weights = []
        self.biases = []
        for i in range(len(sizes) - 1):
            layer_weights = np.random.rand(sizes[i], sizes[i+1]) * 0.02 - 0.01
            self.weights.append(layer_weights)
            layer_biases = np.random.rand(sizes[i+1]) * 0.02 - 0.01
            self.biases.append(layer_biases)

    @classmethod
    def with_dictionary(cls, dict):
        '''
        Accepts a dictionary of parameters and shape and
        initializes a new NeuralNet instance with it.
        '''
        instance = cls(dict["shape"])
        instance.adopt_parameters(dict["parameters"])
        return instance

    def adopt_parameters(self, parameters):
        '''
        Accepts an array of size parameter_size() and applies values
        to the network weights and biases.
        '''
        assert len(parameters) == self.parameter_size()
        index = 0

        parameters = np.array(parameters)

        for layer_weights in self.weights:
            size = layer_weights.size
            new_parameters = np.reshape(parameters[index:index+size], layer_weights.shape)
            layer_weights[0:new_parameters.shape[0], 0:new_parameters.shape[1]] = new_parameters
            index += size

        for layer_biases in self.biases:
            size = layer_biases.size
            new_parameters = parameters[index:index+size]
            layer_biases[0:new_parameters.size] = new_parameters
            index += size

    def parameter_size(self):
        '''
        Returns the number of total parameters in the network
        '''
        s = 0
        for layer_weights in self.weights:
            s += layer_weights.size
        for layer_biases in self.biases:
            s += layer_biases.size
        return s

    def forward(self, nn_input):
        '''
        Forward neural network pass
        '''
        a = nn_input
        for i in range(len(self.weights)):
            z = a.dot(self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                a = nonlinear(z)
            else:
                a = z #linear output layer
        return a

def nonlinear(s):
    '''
    ReLU
    '''
    return np.maximum(0, s)