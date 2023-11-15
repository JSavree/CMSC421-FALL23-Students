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
from Model.layers.bias import BiasLayer
from Model.layers.output_layer import OutputLayer
from Model.optimizers.sgd import SGDSolver
from Model.optimizers.adam import AdamSolver
from Data.data import Data
from Data.generator import q2_b
from Model.evaluate.evaluate import evaluate_model

Number_of_iterations = 5000 # 20000 is too many iterations, I should be around 2000 iterations?
learning_rate = 0.02
# From my experimenting, the first thing I usually tune is the learning rate
# q2_b_hu1000_lr02_5000iter_comparison_plot_1
# Takes about half an hour
# I should focus on experimentation
# report the results of testing the various hyperparameter values, and then explain
# why those results may be happening. Basically, try and synthesize my results.

class Network(BaseNetwork):
    # TODO: you might need to pass additional arguments to init for prob 2, 3, 4 and mnist
    def __init__(self, data_layer, hidden_units):
        # you should always call __init__ first
        super().__init__()
        # TODO: define your network architecture here
        data = data_layer.forward()
        self.input_layer = InputLayer(data_layer)
        self.hidden_layer1 = HiddenLayer(input_dimension=self.input_layer, output_dimension=hidden_units)
        self.bias_layer1 = BiasLayer(input_layer=self.hidden_layer1, activation="ReLU")
        self.output_layer1 = OutputLayer(input_layer=self.bias_layer1, num_out_features=1)
        # TODO: always call self.set_output_layer with the output layer of this network (usually the last layer)
        self.set_output_layer(self.output_layer1)


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
        # TODO: construct your network here
        network = Network(data_layer, hidden_units)
        return network

    def net_setup(self, training_data):
        features, labels = training_data
        # TODO: define input data layer
        self.data_layer = Data(features)
        # TODO: construct the network. you don't have to use define_network.
        # increase hidden units
        self.network = self.define_network(self.data_layer, parameters={'hidden_units': 800})
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
        loss = trainer.train(Number_of_iterations)
        plt.plot(loss)
        plt.ylabel('Loss of NN')
        plt.xlabel('Number of Iterations')
        plt.show()

        # Now let's use the test data
        x_test, y_test = data['test']
        test_data_layer = Data(x_test)
        network.input_layer = InputLayer(test_data_layer)
        network.hidden_layer1.input_layer = network.input_layer

        # Get predictions for test data
        y_pred = network.output_layer1.forward()

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