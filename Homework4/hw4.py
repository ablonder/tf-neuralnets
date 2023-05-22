# hw4.py
# Knowledge Networks and Neuroscience homework #4
# Aviva Blonder

from neuralnet import *
import tflowtools
import numpy as np

def model(data, epochnum, hidnum, testnum, lrate, seed = None):
    """ Trains the network on the data for a given number of epochs and then tests it on a subset of the data
    to create a hinton plot of the activations and weights of each layer """

    # if a seed is provided, seed random
    if seed:
        np.random.seed(seed)
    # shuffle the data
    np.random.shuffle(data)
    # create a neural net object and build the structure of the network
    network = NeuralNet([len(data[0][0])] + hidnum + [len(data[0][1])], lrate)
    # train the network on the data
    network.train(epochs = epochnum, traind = data, validp = .1, validint = 100)
    # shuffle the data again
    np.random.shuffle(data)
    # set the grab variables
    grabacts = [network.net[0]]
    grabwgts = []
    grabbias = []
    for layer in range(1, len(network.net)):
        grabacts.append(network.net[layer]["outputs"])
        grabwgts.append(network.net[layer]["weights"])
        grabbias.append(network.net[layer]["biases"])
    # test on the first testnum instances in the data set to get the activation of each layer
    activations = network.test(data[:testnum], grab = grabacts)[:-1]
    # just test on the first instance to get the weights
    weights = network.test([data[0]], grab = grabwgts)[:-1]
    # and bias
    biases = network.test([data[0]], grab = grabbias)[:-1]
    # create a hinton plot of the neurons, weights, and biases of each layer
    for layer in range(len(activations)):
        tflowtools.hinton_plot(activations[layer])
        plt.title("Layer " + str(layer) + " Activations")
        plt.show()
        # if this is the last hidden layer, display its activations as a dendrogram
        if layer == len(activations)-2:
            tflowtools.dendrogram(activations[layer], np.array(data[:testnum])[:, 0])
            plt.show()
        # if this isn't the input layer, display weights and biases too
        if layer < len(activations)-1:
            tflowtools.hinton_plot(weights[layer])
            plt.title("Layer " + str(layer+1) + " Incoming Weights")
            plt.show()
            tflowtools.display_matrix(weights[layer])
            plt.title("Layer " + str(layer+1) + " Incoming Weights")
            plt.show()
            tflowtools.hinton_plot(np.array(biases[layer]))
            plt.title("Layer " + str(layer+1) + " Biases")
            plt.show()
            tflowtools.display_matrix(np.array(biases[layer]))
            plt.title("Layer " + str(layer+1) + " Biases")
            plt.show()
    # close the session to prevent leakage
    network.sess.close()
    

def main():
    """ This is the only function you have to interact with. Change the values of variables in here to change the parameters to the model function. """

    # generates 100 vectors of length 25 with between 0 and 8 segments
    d = tflowtools.gen_segmented_vector_cases(25, 100, 0, 8)
    # the number of epochs the network will be trained for (train longer for better performance)
    epochs = 5000
    # the number of instances used in testing
    testn = 20
    # a list of the sizes of the hidden layers
    hidn = [3, 3]
    # learning rate parameter
    lrate = .05
    # the actual call to the model function above
    model(d, epochs, hidn, testn, lrate)


main()
