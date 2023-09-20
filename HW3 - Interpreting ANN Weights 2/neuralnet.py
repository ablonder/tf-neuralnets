# neuralnet.py
# generalizable neural net framework (so I don't have to type everything in all the time)
# Aviva Blonder

import tensorflow as tf
import numpy as np
import math
import random
from tffuncs import *

class NeuralNet():

    def __init__(self, layers, lrate):
        """ take a list containing the number of neurons in each layer and the learning rate, and save them as instance variables """
        
        self.struct = layers
        self.lrate = lrate
        # session for use later
        self.sess = None
        # build the network
        self.buildnet(layers)


    def buildnet(self, layers):
        """ build the neural network in tensorflow and save it in the net instance variable """
        
        self.net = [tf.placeholder(tf.float64, shape = (None, layers[0]), name ='input')]
        for l in range(1, len(layers)):
            # create a dictionary to hold the layer
            layer = {}
            # create a weight matrix from the previous layer to the current one
            layer["weights"] = tf.Variable(np.random.uniform(-.1, .1, size = (layers[l-1], layers[l])), name = "w" + str(l))
            # create the biases for this layer
            layer["biases"] = tf.Variable(np.random.uniform(-.1, .1, size = (1, layers[l])), name = "b" + str(l))
            # calculate the output from this layer
            if l == 1:
                layer["outputs"] = tf.nn.relu(tf.matmul(self.net[0], layer["weights"]) + layer["biases"], name = "o" + str(l))
            elif l == len(layers)-1:
                layer["outputs"] = tf.nn.softmax(tf.matmul(self.net[-1]["outputs"], layer["weights"]) + layer["biases"], name = "o" + str(l))
            else:
                layer["outputs"] = tf.nn.relu(tf.matmul(self.net[-1]["outputs"], layer["weights"]) + layer["biases"], name = "o" + str(l))
            # append the layer to the network
            self.net.append(layer)
        # create a placeholder to contain the targets and save as an instance variable
        self.targets = tf.placeholder(tf.float64, shape = (None, layers[-1]), name = "target")
        # save the error as an instance variable
        self.error = tf.reduce_mean(tf.square(self.targets-self.net[-1]["outputs"]), name = "error")
        # create the optimizer
        optimizer = tf.train.GradientDescentOptimizer(self.lrate, name = "bp")
        # use the optimizer to minimize error (and save it as an instance variable)
        self.optimize = optimizer.minimize(self.error)


    def loaddata(self, traindata = None, testprop = None, testdata = None, validprop = None, validdata = None):
        """ Enables the user to load data sets for trianing, testing, and validation. The user also has an option to designate a proportion for validation to split the training set """

        # if testdata has been designated, save it
        if testdata:
            self.testdset = testset
        # if validdata has been designated, save it as the validation set
        if validdata:
            self.validset = validdata
        # if traindata has been designated shuffle it and then save it as is, or split it then save it
        if traindata:
            random.shuffle(traindata)
            # save that as the training set
            self.trainset = traindata
            # if a validation proportion has been designated, split traindata into validation and training sets
            if validprop:
                validsize = int(len(traindata)*validprop)
                # first designate the validation set
                self.validset = traindata[:validsize]
                # then the training set
                self.trainset = traindata[validsize:]
            # if a testing proportion has been designated, split the trainset into test and training sets
            if testprop:
                # figure out how big the test set should actually be
                testsize = int(len(traindata)*testprop)
                # split it off from the trainset
                self.testset = self.trainset[:testsize]
                # save the trainset as the rest
                self.trainset = self.trainset[testsize:]
        
    
    def train(self, grab = [], epochs = 100, batchsize = 1, traind = None, validd = None, validp = None, validint = None):
        """ adjust the weights in the network according to a previously loaded data set, given a list of grabvariables, the number of training epochs, and minibatch size.
    Also gives the user the opportunity to designate a training set and validation set, or validation proportion, and a validation interval. """

        # if no session has been initialized, create a session and save as an instance variable
        if not self.sess:
            self.sess = initsess()
            # also initialize list of training and validation error over epochs of training
            self.trainerror = []
            self.validerror = []

        # loads the data provided
        self.loaddata(traindata = traind, validdata = validd, validprop = validp)

        # figure out how many batches to make and deal with any leftovers
        if batchsize > len(self.trainset):
            # if the designated batch size is larger than the data set, just make the whole data set one batch
            batchnum = 1
            batchsize = len(self.trainset)
            leftovers = 0
        else:
            batchnum = int(len(self.trainset)/batchsize)
            # if the number of batches doesn't divide evenly into the training set, spread out the leftovers
            leftovers = len(self.trainset)%batchnum

        # designate grab vairables
        grabvars = [self.error] + grab
        
        # train the network on the trainset for the designated number of epochs
        for e in range(epochs):
            error = 0
            # start of each batch
            batchs = 0
            # loop through all of the minibatches in trainset and train on each of them
            for batch in range(batchnum):
                # if the current batch is less than the number of leftovers, add one of them to this minibatch
                if batch < leftovers:
                    mbatch = self.trainset[batchs:batchs+batchsize+1]
                    # add 1 to batch start
                    batchs += 1
                # otherwise, just create the minibatch normally
                else:
                    mbatch = self.trainset[batchs:batchs+batchsize]
                # split into inputs
                inputs = [inst[0] for inst in mbatch]
                # and targets
                targets = [inst[1] for inst in mbatch]
                # create a feeder dictionary
                feeder = {self.net[0]: inputs, self.targets: targets}
                # run the sesson on the data and get the grab variables
                grabvals = self.sess.run([self.optimize, grabvars], feed_dict = feeder)[1]
                # add to the error for the session
                error += grabvals[0]
                # increment batch start
                batchs += batchsize
            # add average total error over all batches to the list
            self.trainerror.append(error/batchnum)
            # if a validation interval has been designated, check to see if it's time to do validation
            if validint and e%validint == 0:
                self.validerror.append(self.test(data = self.validset))

        # plot the change in error over the course of training
        # if I ran validation, plot that too
        if validint:
            plt.plot(np.arange(len(self.trainerror)), self.trainerror, "r", np.arange(0, len(self.validerror)*validint, validint), self.validerror, "b")
            plt.legend(["training error", "validation error"])
        else:
            plt.plot(np.arange(len(self.trainerror)), self.trainerror, "r")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Error")
        plt.show()


    def test(self, data = None, grab = []):
        """ tests the network on the provided data """

        # if no data has been designated, just use the test set
        if not data:
            data = self.testset

        # add error to the list of grab variables
        grab.append(self.error)

        # designate inputs
        inputs = [inst[0] for inst in data]
        # and targets
        targets = [inst[1] for inst in data]
        # create a feeder dictionary
        feeder = {self.net[0]: inputs, self.targets: targets}
        # return the designated grab variables and error
        return self.sess.run(grab, feed_dict = feeder)
