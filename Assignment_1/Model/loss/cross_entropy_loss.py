import numpy as np

class CrossEntropyLoss:
    """
    Implements the cross-entropy loss function, commonly used in classification tasks.

    The cross-entropy loss for binary classification is defined as: 
    \( - (y \log(p) + (1 - y) \log(1 - p)) \)

    Attributes:
        input_layer: The preceding layer of the neural network.
        labels: Ground truth labels.
        
    Methods:
        set_data(labels): Method to set the labels.
        forward(): Computes the forward pass, calculating the cross-entropy loss.
        backward(): Computes the backward pass, calculating the gradient of the loss.
    """

    def __init__(self, input_dimension, labels=None) -> None:
        self.input_layer = input_dimension
        self.labels = labels
    
    def set_data(self, labels):
        self.labels = labels

    def forward(self):
        # TODO: Implement the forward pass to compute the loss.
        self.in_array = self.input_layer.forward()
        self.num_data = self.in_array.shape[1]
        # TODO: Compute the result of mean squared error, and store it as self.out_array
        # # not mean squared error, it should be cross entropy loss, right?
        # print(self.num_data)
        # print(self.in_array)
        term_0 = (1 - self.labels) * np.log(1 - self.in_array + 1e-7)
        term_1 = self.labels * np.log(np.abs(self.in_array) + 1e-7) # does log slow things down?
        # self.out_array = (-1 / self.num_data) * (np.dot(self.labels, np.log(self.in_array).T) + np.dot(1 - self.labels, np.log(1 - self.in_array).T))
        self.out_array = -np.mean(term_1 + term_0, axis=0)
        return self.out_array


    def backward(self):
        """
        """
        # TODO: Compute grad of loss with respect to inputs, and hand this gradient backward to the layer behind
        input_grad = (self.in_array - self.labels) / (self.in_array * (1 - self.in_array))
        self.input_layer.backward(input_grad)
        pass



