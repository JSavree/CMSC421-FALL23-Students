from Model.layers.network import BaseNetwork
import numpy as np
import matplotlib.pyplot as plt
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
Number_of_iterations_3layers = 5000 # Experiment to pick your own number of ITERATIONS = batch size
Number_of_iterations_5layers = 2000
# The number of iterations were reduced by a half (from 10000 to 5000), though the
# learning rate was adjusted to 0.005.
Step_size = 0.005 # Experiment to pick your own STEP number = learning rate

class Network(BaseNetwork):
    # TODO: you might need to pass additional arguments to init for prob 2, 3, 4 and mnist
    def __init__(self, data_layer, hidden_units, hidden_layers):
        # you should always call __init__ first
        super().__init__()
        # TODO: define your network architecture here
        data = data_layer.forward()
        self.input_layer = InputLayer(data_layer)
        print("data shape in network", data.shape)
        self.hidden_layer1 = HiddenLayer(self.input_layer, hidden_units[0])
        # For prob 3 and 4:
        # layers.ModuleList can be used to add arbitrary number of layers to the network
        # e.g.:
        # self.MY_MODULE_LIST = layers.ModuleList()
        # for i in range(N):
        #     self.MY_MODULE_LIST.append(layers.Linear(...))
        self.MY_MODULE_LIST = ModuleList()
        self.MY_MODULE_LIST.append(BiasLayer(self.hidden_layer1, activation="ReLU"))
        for i in range(hidden_layers-1):
            self.MY_MODULE_LIST.append(HiddenLayer(self.MY_MODULE_LIST[-1], hidden_units[i+1]))
            self.MY_MODULE_LIST.append(BiasLayer(self.MY_MODULE_LIST[-1], activation="ReLU"))

        self.output_layer1 = OutputLayer(self.MY_MODULE_LIST[-1], 1)
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
        hidden_layers = parameters["hidden_layers"]  # needed for prob 3, 4,
        # TODO: construct your network here
        network = Network(data_layer, hidden_units, hidden_layers)
        return network

    def net_setup(self, training_data, hidden_units, hidden_layers):
        x, y = training_data
        # TODO: define input data layer
        self.data_layer = Data(x)
        # TODO: construct the network. you don't have to use define_network.
        params = {"hidden_units" : hidden_units, "hidden_layers" : hidden_layers}
        self.network = self.define_network(self.data_layer, params)
        # TODO: use the appropriate loss function here
        self.loss_layer = SquareLoss(self.network.get_output_layer(), labels=y)
        # TODO: construct the optimizer class here. You can retrieve all modules with parameters (thus need to be optimized be the optimizer) by "network.get_modules_with_parameters()"
        # change to adam
        self.optimizer = AdamSolver(learning_rate=Step_size, modules=self.network.get_modules_with_parameters())
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
        for iter in range(num_iter):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            print("Training Step: ", iter)

        # you have to return train_losses for the function
        return train_losses

def plot_graph(dataset, pred_line=None, plot_title_num=1, losses=None):
    plots = 2 if losses != None else 1

    fig = plt.figure(figsize=(8 * plots, 6))

    X, y = dataset['X'], dataset['y']

    ax1 = fig.add_subplot(1, plots, 1)

    scatter1 = ax1.scatter(X, y, alpha=0.8)  # Plot the original set of datapoints

    if (pred_line != None):

        x_line, y_line = pred_line['x_line'], pred_line['y_line']

        scatter2 = ax1.scatter(x_line, y_line, color='red', alpha=0.8)
        ax1.legend([scatter1, scatter2], ['Actual', 'Predicted'])
        ax1.set_title('Predicted Line on set of Datapoints plot {}'.format(plot_title_num))

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
def plot_pred_line(X, y, y_pred, plot_title_num, losses=None):
    x_line = X
    y_line = y_pred

    plot_graph(dataset={'X': X, 'y': y}, pred_line={'x_line': x_line, 'y_line': y_line}, plot_title_num=plot_title_num)

    return

# DO NOT CHANGE THE NAME OF THIS FUNCTION
def main(test=False):
    # setup the trainer
    trainer = Trainer()

    data = q2_b()

    # 3 or 5
    is_3_hidden_layers = False;

    # DO NOT REMOVE THESE IF/ELSE
    if not test:
        # Your code goes here.
        # epoch and batch numbers go here.

        # setup network
        # testing with 3 hidden layers:
        if is_3_hidden_layers:
            hidden_units = [128, 64, 32]
            hidden_layers = 3
        else:
            # testing with 5 hidden layers
            hidden_units = [128, 64, 32, 16, 8]
            hidden_layers = 5

        data_layer, network, loss_layer, optimizer = trainer.net_setup(training_data=data['train'], hidden_units=hidden_units, hidden_layers=hidden_layers)

        if is_3_hidden_layers:
            losses = trainer.train(Number_of_iterations_3layers)
        else:
            losses = trainer.train(Number_of_iterations_5layers)

        # Loss plot
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

        plot_pred_line(xtest[:, [0]], ytest, y_preds, 1)
        plot_pred_line(xtest[:, [1]], ytest, y_preds, 2)
        plot_pred_line(xtest[:, [2]], ytest, y_preds, 3)
        plot_pred_line(xtest[:, [3]], ytest, y_preds, 4)
        plot_pred_line(xtest[:, [4]], ytest, y_preds, 5)

        pass
    else:
        # DO NOT CHANGE THIS BRANCH!
        pass


if __name__ == "__main__":
    main()
    pass