from Model.layers.network import BaseNetwork
import numpy as np
from typing import List
import collections.abc
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from Model.layers.input import InputLayer
from Model.layers.hidden import HiddenLayer
from Model.loss.square_loss import SquareLoss
from Model.loss.cross_entropy_loss import CrossEntropyLoss
from Model.layers.bias import BiasLayer
from Model.layers.output_layer import OutputLayer
from Model.layers.network import ModuleList
import Model.layers as layers
from Model.optimizers.sgd import SGDSolver
from Model.optimizers.adam import AdamSolver
from Data.data import Data
from Data.generator import q2_b
from Model.evaluate.evaluate import evaluate_model

# When he says logistic activation function, he means sigmoid


# I started with a default value of hidden units equal to 128, and halved the number of
# hidden units for each layer.
# I also reduced my number of iterations to half of what I did for q2_a, since
# I expected more layers (3 layers used in this case)to take a lower number of iterations
# to reach a good result.
# When using 5 layers, I halved the number of iterations once again (but still keeping
# the same learning rate), and still got a good result from the network. So, it seems
# like increasing the number of layers allows me to decrease the number of iterations by
# a lot, making training much faster.

# I increased the number of iterations from 4500 to 5000 because the R-squared value wasn't as high as
# I'd like (just barely 0.9).
# Like in q3_a, with 5 layers I could get a better or equivalent result with less iterations
# than with 3 layers (5000 vs 2800)
is_3_hidden_layers = True
Number_of_iterations_3layers = 3000 # Experiment to pick your own number of ITERATIONS = batch size
Number_of_iterations_5layers = 2800
# The number of iterations were reduced by a half (from 10000 to 5000), though the
# learning rate was adjusted to 0.005.
learning_rate = 0.01 # Experiment to pick your own STEP number = learning rate

# trial and error is why I'm trying different hyperparameters.
#

class Network(BaseNetwork):
    # TODO: you might need to pass additional arguments to init for prob 2, 3, 4 and mnist
    def __init__(self, data_layer, hidden_units, hidden_layers):
        # you should always call __init__ first
        super().__init__()
        # TODO: define your network architecture here
        data = data_layer.forward()

        self.MY_MODULE_LIST = ModuleList()
        self.MY_MODULE_LIST.append(InputLayer(data_layer))
        for i in range(hidden_layers):
            self.MY_MODULE_LIST.append(HiddenLayer(input_dimension=self.MY_MODULE_LIST[-1],
                                                   output_dimension=hidden_units[i]))
            self.MY_MODULE_LIST.append(BiasLayer(input_layer=self.MY_MODULE_LIST[-1], activation="ReLU"))

        self.MY_MODULE_LIST.append(OutputLayer(input_layer=self.MY_MODULE_LIST[-1], num_out_features=1))
        # TODO: always call self.set_output_layer with the output layer of this network (usually the last layer)
        self.set_output_layer(self.MY_MODULE_LIST[-1])


class Trainer:
    def __init__(self):
        pass

    def define_network(self, data_layer, parameters=None):
        '''
        For prob 2, 3, 4:
        parameters is a dict that might contain keys: "hidden_units" and "hidden_layers".
        "hidden_units" specify the number of hidden units for each layer. "hidden_layers" specify number of hidden layers.
        Note: we might be testing if your network code is generic enough through define_network. Your network code can be even more general, but this is the bare minimum you need to support.
        Note: You are not required to use define_network in setup function below, although you are welcome to.
        '''
        hidden_units = parameters["hidden_units"]  # needed for prob 2, 3, 4
        hidden_layers = parameters["hidden_layers"]  # needed for prob 3, 4,
        # TODO: construct your network here
        network = Network(data_layer, hidden_units, hidden_layers)
        return network

    def net_setup(self, training_data):
        features, labels = training_data
        # TODO: define input data layer
        self.data_layer = Data(features)
        # TODO: construct the network. you don't have to use define_network.
        if is_3_hidden_layers:
            hidden_units = [30, 20, 15]
            hidden_layers = 3
        else:
            # testing with 5 hidden layers
            hidden_units = [128, 64, 32, 16, 8]
            hidden_layers = 5
        params = {"hidden_units" : hidden_units, "hidden_layers" : hidden_layers}
        self.network = self.define_network(self.data_layer, parameters=params)
        # TODO: use the appropriate loss function here
        self.loss_layer = SquareLoss(self.network.get_output_layer(), labels=labels)
        # TODO: construct the optimizer class here. You can retrieve all modules with parameters (thus need to be optimized be the optimizer) by "network.get_modules_with_parameters()"
        # change to adam
        self.optimizer = AdamSolver(learning_rate=learning_rate, modules=self.network.get_modules_with_parameters())
        return self.data_layer, self.network, self.loss_layer, self.optimizer

    def train_step(self):
        # TODO: train the network for a single iteration
        # you have to return loss for the function
        loss = self.loss_layer.forward()
        self.loss_layer.backward()
        self.optimizer.step()

        return loss

    def train(self, num_iter):
        train_losses = []
        # TODO: train the network for num_iter iterations. You should append the loss of each iteration to train_losses.
        for _ in tqdm(range(num_iter), desc="Training", leave=True):
            train_losses.append(self.train_step())

        # you have to return train_losses for the function
        return train_losses


def visualize_data(x_test, y_test, y_pred):
    # Number of features
    num_features = x_test.shape[1]

    # For each feature, plot actual vs predicted
    for i in range(num_features):
        # Create a subplot for each feature
        fig = plt.figure(figsize=(8, 4))

        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x_test[:, i], y_test, alpha=0.5, label='Actual')
        ax.scatter(x_test[:, i], y_pred, alpha=0.5, label='Predicted', marker='x')
        ax.set_xlabel(f'Feature_{i + 1}')
        ax.set_ylabel('Target')
        ax.legend()
        ax.set_title(f'Scatter Plot for Feature_{i + 1}')
        plt.tight_layout()
        plt.show()


# DO NOT CHANGE THE NAME OF THIS FUNCTION
def main(test=False):
    # setup the trainer
    trainer = Trainer()

    # DO NOT REMOVE THESE IF/ELSE
    if not test:
        # Your code goes here.
        data = q2_b()
        data_layer, network, loss_layer, optimizer = trainer.net_setup(data['train'])

        # Since I needed to test 3 vs 5 hidden layers, I had to edit this part of the main function
        if is_3_hidden_layers:
            loss = trainer.train(Number_of_iterations_3layers)
        else:
            loss = trainer.train(Number_of_iterations_5layers)

        plt.plot(loss)
        plt.ylabel('Loss of NN')
        plt.xlabel('Number of Iterations')
        plt.show()

        # Now let's use the test data
        x_test, y_test = data['test']
        test_data_layer = Data(x_test)
        network.input_layer = InputLayer(test_data_layer)
        trainer.data_layer.set_data(network.input_layer)
        trainer.network.MY_MODULE_LIST[1].input_layer = network.input_layer

        # Get predictions for test data
        y_pred = network.MY_MODULE_LIST[-1].forward()

        metrics = evaluate_model(y_test, y_pred)
        # Print the metrics for review
        for key, value in metrics.items():
            print(f"{key}: {value}")

        visualize_data(x_test, y_test, y_pred)

    else:
        # DO NOT CHANGE THIS BRANCH! This branch is used for autograder.
        out = {
            'trainer': trainer
        }
        return out


if __name__ == "__main__":
    main()
    pass