"""
define modules of model
"""
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn


class CNNModel(nn.Module):
    """
    Convolutional Neural Network (CNN) for MNIST classification.
    
    Attributes:
        conv: Convolutional layers.
        fc: Fully connected layer for classification.
    """

    def __init__(self, args):
        super(CNNModel, self).__init__()
        """
        Initialize the CNN model with given arguments.
        
        Args:
            args: Arguments containing hyperparameters.
        """
        
        # Define the model architecture here
        # MNIST image input size batch * 28 * 28 (one input channel)
        # I'm not doing MNIST.
        # I'm doing CINIC-10, which are 32 * 32 RGB pixels.

        # TODO:
        # - Define the architecture for the convolutional layers using nn.Sequential.
        # - Utilize the hyperparameters given by the `args` argument (like the number of channels, kernel size, etc.).
        # - Add the necessary convolutional layers (`nn.Conv2d`), activation functions (`nn.ReLU`, etc.), and pooling layers (`nn.MaxPool2d`, etc.) as needed.
        # - Ensure the depth and the sizes of the feature maps after each layer align with the desired architecture.
        # Convolutional Layers
        self.conv = nn.Sequential(
            # 3 convolutional layers
            nn.Conv2d(in_channels=3, out_channels=args.channel_out1, kernel_size=args.k_size), # 3 input channels, since RGB
            nn.ReLU(),
            nn.MaxPool2d(args.pooling_size, stride=args.max_stride),
            nn.Conv2d(in_channels=args.channel_out1, out_channels=args.channel_out2, kernel_size=args.k_size),
            nn.ReLU(),
            nn.MaxPool2d(args.pooling_size, stride=args.max_stride),
            nn.Conv2d(in_channels=args.channel_out2, out_channels=args.channel_out2, kernel_size=args.k_size),
            nn.ReLU(),
            nn.MaxPool2d(args.pooling_size, stride=args.max_stride)
        )
        # The final output has image size of (1, 1, 64)
        # The following does not include the padding term since padding is 0, so it will just disappear
        image_input_size = 32
        result_size_conv1 = np.floor((image_input_size - args.k_size) / args.stride) + 1
        result_size_maxpool1 = np.floor((result_size_conv1 - args.pooling_size) / args.max_stride) + 1
        result_size_conv2 = np.floor((result_size_maxpool1 - args.k_size) / args.stride) + 1
        result_size_maxpool2 = np.floor((result_size_conv2 - args.pooling_size) / args.max_stride) + 1
        result_size_conv3 = np.floor((result_size_maxpool2 - args.k_size) / args.stride) + 1
        result_size_maxpool3 = np.floor((result_size_conv3 - args.pooling_size) / args.max_stride) + 1

        # TODO:
        # - Define the fully connected (dense) layers for the network.
        # - Determine the input dimension to the first fully connected layer. This should be the flattened size of the feature map produced by the last convolutional layer.
        # - Define linear layers (`nn.Linear`) based on the desired number of neurons in the hidden layers and the number of output classes.
        # - Remember to add activation functions (`nn.ReLU`, etc.) in between these linear layers.
        # - Optionally, consider adding dropout layers (`nn.Dropout`) for regularization if needed.
        # Fully Connected Layers
        # 2 fully connected layers
        self.fc1 = nn.Linear(in_features=args.channel_out2 * 1 * 1, out_features=args.fc_hidden1)
        self.dropout = nn.Dropout(args.dropout)
        self.fc2 = nn.Linear(in_features=args.fc_hidden1, out_features=args.fc_hidden2)
        self.fc3 = nn.Linear(in_features=args.fc_hidden2, out_features=10)

    # Feed features to the model
    def forward(self, x):  # default
        """
        Forward pass of the CNN.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            result: Output tensor of shape (batch_size, num_classes)
        """
        # TODO:
        # - Pass the input tensor `x` through the convolutional layers defined in `self.conv`.
        # - Flatten the resulting feature map to make it suitable for the fully connected layers.
        # - Pass the flattened tensor through the fully connected layers (`self.fc`).
        # - Ensure the final output tensor has a shape compatible with the expected number of classes for the classification task.
        # - Consider using an activation function like softmax if needed at the output (especially if the loss function you're planning to use requires it).

        # TODO Feed input features to the CNN models defined above
        # x = x.view(-1, 1, 32, 32) do I need to do this?
        x_out = self.conv(x)
        
        # TODO Flatten tensor code
        x_out = x_out.view(-1, 64 * 1 * 1)
        x_out = F.relu(self.fc1(x_out))
        x_out = self.dropout(x_out)
        x_out = F.relu(self.fc2(x_out))
        x_out = self.dropout(x_out)
        x_out = self.fc3(x_out) # should I add softmax here at this point?
        
        return x_out
