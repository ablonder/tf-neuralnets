# hw3.py
# Knowledge Networks and Neuroscience homework #3
# Aviva Blonder

from neuralnet import *
import tflowtools
import numpy as np

def model(data, epochnum, hidnum, testnum, lrate = .05, seed = None):
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
    network.train(epochs = epochnum, traind = data)
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

    # comment out one of these lines to switch between generating the binary counts data set and the vector count data set
    #d = tflowtools.gen_all_binary_count_cases(4)
    d = tflowtools.gen_vector_count_cases(100, 4)
    # the number of epochs the network will be trained for (train longer for better performance)
    epochs = 1000
    # the number of instances used in testing
    testn = 20
    # the actual call to the model function above
    model(d, epochs, [3], testn)


main()
