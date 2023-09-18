from Model.layers.network import BaseNetwork
import numpy as np
import matplotlib.pyplot as plt
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


Number_of_iterations = 300 # Experiment to pick your own number of ITERATIONS
Step_size = 0.000001 # Experiment to pick your own STEP number

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
        hidden_units = parameters["hidden_units"]  # needed for prob 2, 3, 4
        hidden_layers = parameters["hidden_layers"]  # needed for prob 3, 4,
        # TODO: construct your network here
        network = Network(data_layer)
        return network

    def net_setup(self, training_data):
        x, y = training_data
        # TODO: define input data layer
        self.data_layer = Data(x) # x is the input data, y is the output?
        # TODO: construct the network. you don't have to use define_network.
        self.network = self.define_network(data_layer=self.data_layer)
        # TODO: use the appropriate loss function here
        self.loss_layer = ...
        # TODO: construct the optimizer class here. You can retrieve all modules with parameters (thus need to be optimized be the optimizer) by "network.get_modules_with_parameters()"
        self.optimizer = ...
        return self.data_layer, self.network, self.loss_layer, self.optim

    def train_step(self):
        # TODO: train the network for a single iteration
        # you have to return loss for the function
        # I use a regression loss for training

        loss = ...
        return loss

    def get_num_iters_on_public_test(self):
        # TODO: adjust this number to how much iterations you want to train on the public test dataset for this problem.
        return 30000

    def train(self, num_iter):
        train_losses = []
        # TODO: train the network for num_iter iterations. You should append the loss of each iteration to train_losses.
        # So, I call train step to do the step of training, 


        # you have to return train_losses for the function
        return train_losses


# DO NOT CHANGE THE NAME OF THIS FUNCTION
def main(test=False):
    # setup the trainer
    trainer = Trainer()

    # DO NOT REMOVE THESE IF/ELSE
    if not test:
        # Your code goes here.

        pass
    else:
        # DO NOT CHANGE THIS BRANCH!
        pass


if __name__ == "__main__":
    main()
    pass
