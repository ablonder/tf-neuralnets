# hw2.py
# Knowledge Networks and Neuroscience homework #2
# Aviva Blonder

from neuralnet import *
import tflowtools
import numpy as np

def model(data, epochnum, hidnum, testnum, lrate = .05, seed = None):
    """ Trains the network on the data for a given number of epochs and then tests it on a subset of the data to create a hinton plot and dendrogram of the hidden layer """

    # if a seed is provided, seed random
    if seed:
        np.random.seed(seed)
    # shuffle the data
    np.random.shuffle(data)
    # create a neural net object and build the structure of the network
    network = NeuralNet([len(data[0][0])] + hidnum + [len(data[0][1])], lrate)
    # train the network on the data
    network.train(epochs = epochnum, traind = data)
    # shuffle the data again
    np.random.shuffle(data)
    # set the grab variables
    grabvars = [network.net[0]]
    for layer in range(1, len(network.net)):
        grabvars.append(network.net[layer]["outputs"])
    # test on the first testnum instances in the data set
    results = network.test(data[:testnum], grab = grabvars)
    # create a hinton plot of the neurons of each layer
    for layer in range(len(results)-1):
        tflowtools.hinton_plot(results[layer])
        plt.title("Layer " + str(layer) + " Activations")
        plt.show()
        # and a dendrogram if this is the hidden layer
        if layer > 0 and layer < len(results)-2:
            tflowtools.dendrogram(results[layer], np.array(data[:testnum])[:, 0])

casel = 6
#d = tflowtools.gen_all_binary_count_cases(casel)
d = tflowtools.gen_random_line_cases(100, (casel, casel), mode = 'auto')
epochs = 1000
hidn = 8
testn = 20
model(d, epochs, [hidn], testn)
