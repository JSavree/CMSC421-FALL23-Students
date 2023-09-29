import numpy as np

class SquareLoss:
    """
    Implements the square loss function, commonly used in regression tasks.

    The square loss is defined as: \( \frac{1}{2M} || X - Y ||^2 \)

    Attributes:
        input_layer: The preceding layer of the neural network.
        labels: Ground truth labels.
        num_data: Number of data samples.

    Methods:
        set_labels(labels): Method to set the labels.
        forward(): Computes the forward pass, calculating the square loss.
        backward(): Computes the backward pass, calculating the gradient of the loss.
    """

    def __init__(self, input_dimension, labels=None) -> None:
        self.input_layer = input_dimension
        # TODO: Initialize self.labels and reshape it to (-1, 1)
        self.labels = np.reshape(labels, (-1, 1))
        
    
    def set_labels(self,labels):
        # TODO: Implement code to set self.labels
        self.labels = labels

    def forward(self):
        """Loss value is (1/2N) || X-Y ||^2"""
        # Implement the forward pass to compute the loss.
        self.in_array = self.input_layer.forward()
        self.num_data = self.in_array.shape[0]
        # TODO: Compute the result of mean squared error, and store it as self.out_array
        self.out_array = np.square(np.linalg.norm(self.in_array - self.labels)) * (0.5 * self.num_data) #X and Y should be the same dimension, so
        # I should do l2 norm using numpy, then take the mean and square it
        # this is a scalar? Or do I do
        # self.out_array = np.sum(np.square(self.in_array - self.labels)) / (2 * self.num_data)
        return self.out_array

    def backward(self):
        """
        Gradient is (1/N) (X-Y), where N is the number of training samples
        """
        # TODO: Compute grad of loss with respect to inputs, and hand this gradient backward to the layer behind
        self.pass_back = (self.in_array - self.labels) / self.num_data # this should be an array (matrix), since we
        # want to compute the gradients and stuff this time.
        self.input_layer.backward(self.pass_back)  # hand the gradient of loss with respect to inputs back to previous layer
        pass
    pass