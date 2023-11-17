from Model.layers.network import BaseNetwork
import numpy as np
from typing import List
import collections.abc
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
np.random.seed(42)
from Model.layers.input import InputLayer
from Model.layers.hidden import HiddenLayer
from Model.loss.square_loss import SquareLoss
from Model.layers.bias import BiasLayer
from Model.layers.output_layer import OutputLayer
from Model.optimizers.sgd import SGDSolver
from Model.optimizers.adam import AdamSolver
from Data.data import Data
from Data.generator import q1_a
from Model.evaluate.evaluate import evaluate_model

Number_of_iterations = 3000
learning_rate = 0.05

plots_file_path = "C:/Users/aqwan/GitHub/CMSC421-FALL23-Students/Assignment_1/plots"


class Network(BaseNetwork):
    def __init__(self, data_layer):
        super().__init__()
        data = data_layer.forward()
        self.input_layer = InputLayer(data_layer)
        print("data shape in network", data.shape)
        self.hidden_layer1 = HiddenLayer(self.input_layer, 1)
        self.bias_layer1 = BiasLayer(self.hidden_layer1)
        self.output_layer1 = OutputLayer(self.bias_layer1, 1) # Activation funciton is linear by default
        self.set_output_layer(self.output_layer1)


# To get you started we built the network for you!! Please use the template file to finish answering the question
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
        #hidden_units = parameters["hidden_units"]  # needed for prob 2, 3, 4
        #hidden_layers = parameters["hidden_layers"]  # needed for prob 3, 4,
        # TODO: construct your network here
        network = Network(data_layer)
        return network

    def net_setup(self, training_data):
        x, y = training_data # x is the features, y are the labels for the data
        # TODO: define input data layer
        self.data_layer = Data(x) # x is the input data, y is the output?
        # TODO: construct the network. you don't have to use define_network.
        self.network = self.define_network(data_layer=self.data_layer)
        # TODO: use the appropriate loss function here
        self.loss_layer = SquareLoss(self.network.get_output_layer(), labels=y)
        # TODO: construct the optimizer class here. You can retrieve all modules with parameters (thus need to be optimized be the optimizer) by "network.get_modules_with_parameters()"
        self.optimizer = SGDSolver(learning_rate=learning_rate, modules=self.network.get_modules_with_parameters())
        return self.data_layer, self.network, self.loss_layer, self.optimizer

    def train_step(self):
        # TODO: train the network for a single iteration
        # you have to return loss for the function
        # I use a regression loss for training
        # loss layer = updating gradients
        # optimizer = updating weights and biases

        loss = self.loss_layer.forward()
        self.loss_layer.backward()
        self.optimizer.step()

        return loss

    def train(self, num_iter):
        train_losses = []

        for _ in tqdm(range(num_iter), desc="Training", leave=True):
            train_losses.append(self.train_step())

        return train_losses


# DO NOT CHANGE THE NAME OF THIS FUNCTION
def main(test=False):
    # setup the trainer
    trainer = Trainer()

    # DO NOT REMOVE THESE IF/ELSE
    if not test:
        data = q1_a()
        data_layer, network, loss_layer, optimizer = trainer.net_setup(data['train'])
        loss = trainer.train(Number_of_iterations)
        plt.plot(loss)
        plt.ylabel('Loss of NN')
        plt.xlabel('Number of Iterations')
        file_name = plots_file_path + "/q1_a_lr{}_iters{}_loss_plot.png".format(learning_rate, Number_of_iterations)
        plt.savefig(file_name)
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

        # Plot actual vs predicted on test data
        plt.figure(figsize=(10, 6))
        plt.scatter(x_test, y_test, label='Actual', alpha=0.6)
        plt.scatter(x_test, y_pred, label='Predicted', alpha=0.6)
        plt.xlabel('Feature (x)')
        plt.ylabel('Target (y)')
        plt.title('Test Data and Model Predictions')
        plt.legend()
        file_name = plots_file_path + "/q1_a_lr{}_iters{}_comparison_plot.png".format(learning_rate,
                                                                                         Number_of_iterations)
        plt.savefig(file_name)
        plt.show()

        pass
    else:
        # DO NOT CHANGE THIS BRANCH! This branch is used for autograder.
        out = {
            'trainer': trainer
        }
        return out


if __name__ == "__main__":
    main()
    pass
