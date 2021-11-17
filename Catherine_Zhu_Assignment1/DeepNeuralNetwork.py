import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from collections import deque


def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
   # X, y = datasets.make_blobs(n_samples=200, centers=2, n_features=2)
    return X, y


def plot_decision_boundary(pred_func, X, y, func_type, Number_of_Layers, nn_layersN):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    func_type = str(func_type)
    Number_of_Layers = str(Number_of_Layers)
    Layers = str(nn_layersN)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.title("Type:" + func_type + ", Number of Layers:" + Number_of_Layers + ", Layers:" + Layers)
    plt.show()


########################################################################################################################
########################################################################################################################

class DeepNeuralNetwork(object):
    """
    This class builds and trains a neural network
    """

    def __init__(self, nn_layersN, nn_layersize, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        #        self.nn_input_dim = nn_input_dim
        #        self.nn_hidden_dim = nn_hidden_dim
        #        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        #######
        self.nn_layersN = nn_layersN  # Number of layers
        self.nn_layersize = nn_layersize  # Number of neurons in each layer
        #####

        # initialize the weights and biases in the network
        self.W = []
        self.b = []

        np.random.seed(seed)

        for i in range(self.nn_layersN - 1):
            self.W.append(
                np.random.randn(self.nn_layersize[i], self.nn_layersize[i + 1]) / np.sqrt(self.nn_layersize[i]))
            self.b.append(np.zeros((1, self.nn_layersize[i + 1])))

    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''

        # YOU IMPLMENT YOUR actFun HERE
        if type == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif type == "tanh":
            return np.tanh(z)
        elif type == "ReLu":
            return np.maximum(0, z)
        else:
            return "Unknown activation function"

    def diff_actFun(self, z, type):
        '''
        diff_actFun compute the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''

        # YOU IMPLEMENT YOUR diff_actFun HERE
        if type == "sigmoid":
            ss = 1 / (1 + np.exp(-z))
            return ss * (1 - ss)
        elif type == "tanh":
            return 1 - np.tanh(z) ** 2
        elif type == "ReLu":
            return 1 * (z > 0)
        else:
            return "Unknown derivatives"

    def feedforward(self, X, actFun):
        '''
        feedforward builds a n-layer neural network and computes the probabilitie
        :param X: input data, and further in the loop it becomes the output data of each layer
        :param actFun: activation function
        :return:
        '''

        # YOU IMPLEMENT YOUR feedforward HERE
        self.z = []
        self.a = []

        for i in range(self.nn_layersN - 1):
            if i == 0:
                self.z.append(np.dot(X, self.W[i]) + self.b[i])
            else:
                self.z.append(np.dot(self.a[i - 1], self.W[i]) + self.b[i])

            self.a.append(self.actFun(self.z[i], self.actFun_type))
        self.probs = np.exp(self.z[-1]) / np.sum(np.exp(self.z[-1]), axis=1, keepdims=True)
        return self.probs

    def calculate_loss(self, X, y):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE
        probs = np.exp(self.z[-1]) / np.sum(np.exp(self.z[-1]), axis=1, keepdims=True)
        data_loss = np.sum(-np.log(probs[np.arange(num_examples), y]))

        # Add regulatization term to loss (optional)
        sum_W = 0
        for i in range(self.nn_layersN - 1):
            sum_W += np.sum(np.square(self.W[i]))
        data_loss += self.reg_lambda / 2 * (sum_W)
        return (1 / num_examples) * data_loss

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''
        dW = deque([])
        db = deque([])

        # IMPLEMENT YOUR BACKPROP HERE
        N = len(X)
        delta = np.exp(self.z[-1]) / np.sum(np.exp(self.z[-1]), axis=1, keepdims=True)
        delta[np.arange(N), y] -= 1
        for i in range(self.nn_layersN - 1):
            pos_index = self.nn_layersN - 2 - i
            if pos_index != 0:
                dW.appendleft(np.dot(self.a[pos_index - 1].T, delta))
                db.appendleft(np.sum(delta, axis=0, keepdims=True))
                delta = np.dot(delta, self.W[pos_index].T) * \
                        self.diff_actFun(self.z[pos_index - 1], type=self.actFun_type)
            else:
                dW.appendleft(np.dot(X.T, delta))
                db.appendleft(np.sum(delta, axis=0, keepdims=False))

        return dW, db

    def fit_model(self, X, y, epsilon=0.001, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for j in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation

            dW, db = self.backprop(X, y)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            for i in range(len(dW)):
                dW[i] += self.reg_lambda * self.W[i]

            # Gradient descent parameter update
            for i in range((self.nn_layersN - 1)):
                self.W[i] += -epsilon * dW[i]
                self.b[i] += -epsilon * db[i]

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and j % 100 == 0:
                print("Loss after iteration %i: %f" % (j, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y, func_type, Number_of_Layers, nn_layersN):
        '''
        visualize_decision_boundary plots the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y, func_type, Number_of_Layers, nn_layersN)


def main():
    # generate and visualize Make-Moons dataset
    X, y = generate_data()
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    while True:
        try:
            function_type = str(input("Please enter the activation function (choose from tanh, sigmoid, or ReLu):"))
            x_in = int(input("Please Enter Number of Layers:"))
            num_input = [int(s) for s
                         in input("please Enter How many Units in each hidden layer separated by common:").split(',')]
            x_num = num_input

            print(x_num)
        except ValueError:
            choice = str(input("Sorry, I don't understand. Exit, press Q. Try again, press any other keys "))
            if choice == str('Q'): break; break
            else: continue
        if len(x_num) < x_in-2:
            choice = str(input("We want more layers as input. Exit, press Q. Try again, press any other keys "))
            if choice == str('Q'): break ; break
            else: continue
        else:
            x_final_num = [2, 2]
            pos_index = 1
            index_hidden_layer = int(x_in - 2)
            print("Taking " + str(index_hidden_layer) + " hidden layer.")
            for i in range(index_hidden_layer):
                x_final_num.insert(pos_index, x_num[i])
                pos_index += 1
            print("Number of neurons in each layer: " + str(x_final_num))
            print("Running DeepNeuralNetwork")
            model = DeepNeuralNetwork(nn_layersN=x_in, nn_layersize=x_final_num, actFun_type=function_type)

            model.fit_model(X, y)
            model.visualize_decision_boundary(X, y, function_type, x_in, x_final_num)
            print("All Done!")
            break
if __name__ == "__main__":
    main()
