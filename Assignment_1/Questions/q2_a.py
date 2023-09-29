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
from Data.generator import q2_a
from Model.evaluate.evaluate import evaluate_model

# First tested with 3000 iterations, saw from the loss graph it hit near 0 loss much sooner,
# so I decreased the number of iterations to 100. Saw I didn't need 100 iterations either,
# decreased iterations some more. But the predicated values still looked bad, so I then changed
# my learning rate (decreased it). The line started looking better, but now I needed to increase
# my number of iterations.
# I started with a default value of hidden units equal to 128
Number_of_iterations = 600 # Experiment to pick your own number of ITERATIONS = batch size
Step_size = 1e-6 # Experiment to pick your own STEP number = learning rate
n_epochs = 100

class Network(BaseNetwork):
    # TODO: you might need to pass additional arguments to init for prob 2, 3, 4 and mnist
    def __init__(self, data_layer, hidden_units):
        # you should always call __init__ first
        super().__init__()
        # TODO: define your network architecture here
        data = data_layer.forward()
        self.input_layer = InputLayer(data_layer)
        print("data shape in network", data.shape)
        self.hidden_layer1 = HiddenLayer(self.input_layer, hidden_units)
        self.bias_layer1 = BiasLayer(self.hidden_layer1, activation="ReLU")
        self.output_layer1 = OutputLayer(self.bias_layer1, 1)
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
        #hidden_layers = parameters["hidden_layers"]  # needed for prob 3, 4,
        # TODO: construct your network here
        network = Network(data_layer, hidden_units)
        return network

    def net_setup(self, training_data, hidden_units):
        x, y = training_data
        # TODO: define input data layer
        self.data_layer = Data(x)
        # TODO: construct the network. you don't have to use define_network.
        params = {"hidden_units" : hidden_units}
        self.network = self.define_network(self.data_layer, params)
        # TODO: use the appropriate loss function here
        self.loss_layer = SquareLoss(self.network.get_output_layer(), labels=y)
        # TODO: construct the optimizer class here. You can retrieve all modules with parameters (thus need to be optimized be the optimizer) by "network.get_modules_with_parameters()"
        self.optimizer = SGDSolver(learning_rate=Step_size, modules=self.network.get_modules_with_parameters())
        return self.data_layer, self.network, self.loss_layer, self.optimizer

    def train_step(self):
        # TODO: train the network for a single iteration
        # you have to return loss for the function
        loss = self.loss_layer.forward()

        self.loss_layer.backward()

        self.optimizer.step()

        return loss

    def get_num_iters_on_public_test(self):
        # TODO: adjust this number to how much iterations you want to train on the public test dataset for this problem.
        return 30000

    def train(self, num_iter):
        train_losses = []
        # TODO: train the network for num_iter iterations. You should append the loss of each iteration to train_losses.
        for iter in range(num_iter):
            train_loss = self.train_step()
            train_losses.append(train_loss)

        # you have to return train_losses for the function
        return train_losses

def plot_graph(dataset, pred_line=None, losses=None):
    plots = 2 if losses != None else 1

    fig = plt.figure(figsize=(8 * plots, 6))

    X, y = dataset['X'], dataset['y']

    ax1 = fig.add_subplot(1, plots, 1)

    # y_sort = np.sort(y, axis=0)

    scatter1 = ax1.scatter(X, y, alpha=0.8)  # Plot the original set of datapoints

    if (pred_line != None):

        x_line, y_line = pred_line['x_line'], pred_line['y_line']

        # ax1.plot(x_line, y_line, linewidth=2, markersize=12, color='red', alpha=0.8)  # Plot the randomly generated line
        scatter2 = ax1.scatter(x_line, y_line, color='red', alpha=0.8)
        ax1.legend([scatter1, scatter2], ['Actual', 'Predicted'])
        ax1.set_title('Predicted Line on set of Datapoints')

    else:
        ax1.set_title('Plot of Datapoints generated')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    if (losses != None):
        ax2 = fig.add_subplot(1, plots, 2)
        ax2.plot(np.arange(len(losses)), losses, marker='o')

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss')

    plt.show()


# Function to plot predicted line

def plot_pred_line(X, y, y_pred, losses=None):
    # Generate a set of datapoints on x for creating a line.
    # We shall consider the range of X_train for generating the line so that the line superposes the datapoints.
    x_line = X
    # Calculate the corresponding y with the parameter values of m & b
    y_line = y_pred # np.sort(y_pred, axis=0)

    plot_graph(dataset={'X': X, 'y': y}, pred_line={'x_line': x_line, 'y_line': y_line})

    return

# DO NOT CHANGE THE NAME OF THIS FUNCTION
# DO NOT CHANGE THE NAME OF THIS FUNCTION
def main(test=False):
    # setup the trainer
    trainer = Trainer()

    data = q2_a()

    # DO NOT REMOVE THESE IF/ELSE
    if not test:
        # Your code goes here.
        # epoch and batch numbers go here.

        # setup network
        data_layer, network, loss_layer, optimizer = trainer.net_setup(training_data=data['train'], hidden_units=64)
        losses = trainer.train(Number_of_iterations)

        plt.plot(losses)
        plt.ylabel("Loss of Neural Network")
        plt.xlabel("Number of Iterations (Epochs)")
        plt.show()

        xtest, ytest = data["test"]
        network.input_layer = InputLayer(Data(xtest))
        network.hidden_layer1.input_layer = network.input_layer

        y_preds = network.output_layer1.forward()

        metrics = evaluate_model(ytest, y_preds)
        print(metrics)

        # start = 0
        # end = 10
        # step_sizes = (end - start) / len(ytest)
        # x_values = np.arange(start, end, step_sizes)

        plot_pred_line(xtest, ytest, y_preds, losses)


        pass
    else:
        # DO NOT CHANGE THIS BRANCH!
        pass


if __name__ == "__main__":
    main()
    pass