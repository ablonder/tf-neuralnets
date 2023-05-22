# tffuncs.py
# Useful functions for running tensorflow more easily and efficiently
# Aviva Blonder

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# initialize a session
def initsess(direct = None):
    sess = tf.Session()
    if direct:
        # if a directory has been designated, log the session for later viewing
        sess.probe_stream = viewprep(sess, directory = direct)
        sess.viewdir = direct
    # initialize all variables at once
    sess.run(tf.global_variables_initializer())
    # return the session
    return sess
    
# runs a session on the provided operators, and returns the result, also visualizes the session on the tensorboard
def quickrun(operators, grabvars = [], direct = None, feeddict = None, session = None):
    # if necessary, initialize a session
    if not session:
        session = initsess(direct)
    # run the session on the operators, grab variables, and a feed dictionary if applicable
    if feeddict:
        result = session.run([operators, grabvars], feed_dict = feeddict)
    else:
        result = session.run([operators, grabvars])
    return result

# if we don't want to use the session any more, close it and if a directory was provided, visualize it on the tensorboard
def endsess(session, direct = None):
    session.close()
    if direct:
        firetensorboard(session.viewdir)

# set up the tensorboard so we can visualize the function graph - doesn't like that I'm using a keyword below
#def veiwprep(session, directory = 'tfview', flush = 120, queue = 10):
    # create a file writer object that puts the information into a file, to be used to make the tensorboard
    #return tf.summary.FileWriter(directory.session.graph.flush_secs = flush, max.queue = queue)

# runs the graph from the command line
def firetensorboard(logdir):
    os.system("tensorboard --logdir=" + logdir)
    # then open it up in a browser by going to localhost:6006
