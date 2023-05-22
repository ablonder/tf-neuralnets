# hw1.py
# Knowledge Networks and Neuroscience, homework assignment #1.
# Aviva Blonder

import neuralnet
import tflowtools as tft
import numpy as np

def autoencoder(hidnum, e):
    """ Produces a graph of the error of a neural network trained to reproduce one-hot patterns of length 2^hidnum. """

    # all of the possible one hot vectors of length 2^hidnum which will serve as the training set
    data = tft.gen_all_one_hot_cases(2^hidnum)
    # create a neural network with input and output layers of size 2^hidnum and one hidden layer with hidnum neurons
    net = neuralnet.NeuralNet([2^hidnum, hidnum, 2^hidnum], .03)
    # train the neural network on the data for the given number of epochs
    net.train(epochs = e, traind = data)


def parity(insize, e, hidlayers = [10]):
    """ Produces a graph of the error of a neural network trained to output whether there is an even or odd number of ones in a binary vector of size insize. """

    # all of the possible binary vectors of length insize which will serve as the training/validation set
    data = tft.gen_all_parity_cases(insize, double = True)
    # create a neural network with an input layer of size insize, output layer of size two, and the provided hidden layers
    net = neuralnet.NeuralNet([insize] + hidlayers + [2], .03)
    # train the neural network on the data for the given number of epochs, using 10% as a validation set, with validation every 10 epochs
    net.train(epochs = e, traind = data, validp = .1, validint = 10)


def winequality(lrate, e, validp, testp, hidlayers = [10], vint=10):
    """ Produces a graph of the error of a neural network trained to categorize wines """

    # load the wine quality data set
    wines = np.loadtxt("wine_quality.txt", delimiter = ";")
    # split it into attributes and labels - firgure out how to make each of these a list
    wdata = []
    # generate all one hot cases of size 9 for the targets
    targets = tft.gen_all_one_hot_cases(9)
    for inst in wines:
        wdata.append([inst[:11], targets[int(inst[11])-1][0]])
    # create a neural network with an input layer of 11 nodes, a hidden layer of the provided number of nodes and an output layer of one node, and the designated lrate
    net = neuralnet.NeuralNet([11] + hidlayers + [9], lrate)
    # load wdata and split it into training, test, and validation sets according to validprop and testprop
    net.loaddata(traindata = wdata, testprop = testp, validprop = validp)
    # train the network with validation every 10 epochs
    net.train(epochs = e, validint = vint)
    # test the network and print out the resulting error
    print(net.test())

def letterrecog(lrate, e, validp, testp, hidlayers = [10], vint=10):
    """ Produces a graph of the error of a neural network trained to categorize letters based on various features """

    # load the wine quality data set
    letters = np.loadtxt("letter-recognition.csv", delimiter = ",", converters={0: lambda x: ord(x)-ord('A')})
    # split it into attributes and labels - firgure out how to make each of these a list
    ldata = []
    # generate all one hot cases of size 26 for the targets
    targets = tft.gen_all_one_hot_cases(26)
    for inst in letters:
        ldata.append([inst[1:], targets[int(inst[1])][0]])
    # create a neural network with an input layer of 11 nodes, a hidden layer of the provided number of nodes and an output layer of one node, and the designated lrate
    net = neuralnet.NeuralNet([16] + hidlayers + [26], lrate)
    # load wdata and split it into training, test, and validation sets according to validprop and testprop
    net.loaddata(traindata = ldata, testprop = testp, validprop = validp)
    # train the network with validation every 10 epochs
    net.train(epochs = e, validint = vint)
    # test the network and print out the resulting error
    print(net.test())


autoencoder(4, 5000)
