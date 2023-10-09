import numpy as np


def sigmoid_func(x):
    # Prevent overflow.
    outs = np.zeros_like(x)
    outs[x < 0] = np.exp(x[x < 0]) / (1 + np.exp(x[x < 0]))
    outs[x > 0] = 1 / (1 + np.exp(-x[x > 0]))
    return outs


class Sigmoid:
    """
    Implements the Sigmoid activation function.
    
    The sigmoid function is defined as: f(x) = 1 / (1 + e^{-x})
    
    Attributes:
        input_layer: The layer that provides the input to this activation function.
        
    Methods:
        forward(): Applies the Sigmoid activation function to the output of the input layer.
        backward(downstream): Computes the gradient of the loss with respect to the input, which is then passed back to the previous layers.
    """


    @staticmethod
    def forward(input_array):
        # Apply the Sigmoid activation function to the output of the input layer
        output_array = sigmoid_func(input_array)
        return output_array
    
    @staticmethod
    def backward(downstream, input_array=None):
        # Compute the gradient of the loss with respect to the input
        # do downstream * gradient (which is computed using input array)
        sig_grad = sigmoid_func(input_array) * (1 - sigmoid_func(input_array))
        input_grad = downstream * sig_grad
        return input_grad



