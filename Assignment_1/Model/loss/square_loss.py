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
        """Loss value is (1/2M) || X-Y ||^2"""
        self.in_array = self.input_layer.forward()
        # Determine the number of data samples.
        self.num_data = self.in_array.shape[0]

        # Implement the calculation of the squared loss:
        # TODO: Subtract 'self.labels' from 'self.in_array' to get the element-wise difference between
        # the predicted values and the ground truth labels.
        difference = self.in_array - self.labels
        # TODO: Use 'np.linalg.norm()' to compute the Euclidean norm (or L2 norm) of the difference.
        # This function returns the Frobenius norm when used on matrices, which effectively
        # calculates the root of the sum of the squared differences across all elements.
        # Square the resulting norm to get the sum of squared differences.
        euclidean_norm = np.square(np.linalg.norm(difference))
        # TODO: Multiply the sum of squared differences in `euclidean_norm` by the results of dividing 0.5 by 'self.num_data'
        # to compute the mean of the squared differences, which represents the average loss
        # across all data samples.
        # Store this result in 'self.out_array'.
        self.out_array = euclidean_norm * (0.5 / self.num_data)

        return self.out_array

    def backward(self):
        """
        Gradient is (1/M) (X-Y), where M is the number of training samples
        """

        # TODO: Compute the element-wise difference between the predicted values (in 'self.in_array')
        # and the actual ground truth values (in 'self.labels'). This difference represents the error
        # for each data sample across all features.
        # To compute the gradient of the squared loss with respect to each prediction, you need to
        # divide each element of this difference by 'self.num_data'. This operation effectively scales
        # down the error by the total number of data samples, giving you the average error gradient
        # across all data samples. The resulting matrix/array will be the gradient of the loss
        # with respect to the outputs of the preceding layer in the neural network. This gradient
        # , termed 'self.pass_back', will be used by preceding layers to adjust their parameters
        # during the backward propagation step.
        self.pass_back = (self.in_array - self.labels) / self.num_data

        # Hand the gradient of loss with respect to inputs back to the previous layer.
        self.input_layer.backward(self.pass_back)
        pass
    pass