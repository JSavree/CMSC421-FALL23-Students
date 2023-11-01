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
        """
        Computes the forward pass, calculating the binary cross-entropy loss.

        TODOs:
        1. Calculate the negative log likelihood for the true labels (y) using the formula.
        2. Average the computed loss over all data points.

        Returns:
        - float: The average binary cross-entropy loss.
        """
        # Fetch the predicted probabilities from the preceding layer's forward
        self.p = self.input_layer.forward()
        # batch size for samples
        self.num_data = self.p.shape[1]

        # TODO 1: Compute the loss using the provided formula for each data point
        # - [y \log(p) + (1 - y) \log(1 - p)]
        loss_values = -((self.labels*np.log(self.p)) + ((1 - self.labels) * np.log(1 - self.p)))

        # TODO 2: Average the computed loss over all data points. You can use np.mean for this.
        self.out_array = np.mean(loss_values)
        return self.out_array

    def backward(self):
        """
        Computes the backward pass, calculating the gradient of the loss with respect to the input.

        The derivative of the binary cross-entropy loss with respect to the predicted probability \( p \) is:
        \( \frac{\partial \text{loss}}{\partial p} = \frac{-y}{p} + \frac{1 - y}{1 - p} \)

        TODOs:
        1. Compute the derivative of the binary cross-entropy loss with respect to the predicted probabilities using the formula.

        Returns:
        - numpy.array: The gradient of the loss with respect to the input.
        """
        # TODO 1. Compute the negative gradient for true labels (-labels / predicted probabilities)
        negative_gradient_for_positives = (-self.labels / self.p)
        # TODO 2. Compute the gradient for false labels ((1 - labels) / (1 - predicted probabilities))
        gradient_for_negatives = ((1 - self.labels) / (1 - self.p))
        # TODO 3. Combine the gradients computed above by subtracting the gradient for true labels from the gradient for false labels and Normalize the gradient by the batch size
        input_grad = (negative_gradient_for_positives + gradient_for_negatives) / self.num_data

        # Return the computed gradient to the preceding layers to continue the chain of gradients
        self.input_layer.backward(input_grad)
        pass



