import numpy as np
class ReLU:
    """
    Implements the Rectified Linear Unit (ReLU) activation function.
    
    The relu function is defined as: f(x) = max(0,x)

    Attributes:
        input_layer: The layer that feeds input into this activation function.
        input_dimension: The shape of the output from the input_layer.
        output_dimension: The shape of the output of this layer, which 
        is the same as the input_dimension for ReLU.
        
    Methods:
        forward(): Applies the ReLU activation function to the output 
        of the input layer.
        backward(downstream): Computes the gradient of the loss with 
        respect to the input, to be passed back to the previous layers.
    """


    @staticmethod
    def forward(input_array):
        #TODO: Apply the ReLU activation function to the output of the input layer
        output_array = np.maximum(0, input_array)
        return output_array
    
    @staticmethod
    def backward(downstream, input_array=None):
        #TODO: Compute the gradient of the loss with respect to the input

        #print("downstream shape", downstream.shape)
        #print("input array", input_array.shape)
        mask = input_array > 0
        input_grad = downstream * mask
        #print("input grad shape", input_grad.shape)
        return input_grad
