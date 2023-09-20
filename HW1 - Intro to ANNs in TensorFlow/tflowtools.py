# The functions in this file are used to generate datasets for machine-learning problems.

import tensorflow as tf
import numpy as np
import os  # For starting up tensorboard from inside python
import matplotlib.pyplot as PLT

# ****** SESSION HANDLING *******

def gen_initialized_session(dir='probeview'):
    sess = tf.Session()
    sess.probe_stream = viewprep(sess,dir=dir)  # Create a probe stream and attach to the session
    sess.viewdir = dir  # add a second slot, viewdir, to the session
    sess.run(tf.global_variables_initializer())
    return sess

def copy_session(sess1):
    sess2 = tf.Session()
    sess2.probe_stream = sess1.probe_stream
    sess2.viewdir = sess1.viewdir
    return sess2

def close_session(sess, view=True):
    sess.close()
    if view: fireup_tensorboard(sess.viewdir)

# Simple evaluator of a TF operator.
def tfeval(operators):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    result = sess.run(operators) # result = a list of output values, one from each operator.
    sess.close()
    return result

# ***** TENSORBOARD SUPPORT ****

# This creates the main data for tensorboard viewing: the graph and variable histories.

def viewprep(session, dir='probeview',flush=120,queue=10):
    return tf.summary.FileWriter(dir,session.graph,flush_secs=flush,max_queue=queue)

# To view probes, the function graph, etc., do this at the command line:
#        tensorboard --logdir=probeview
# Then open a Chrome browser and go to site:  localhost:6006

def fireup_tensorboard(logdir):
    os.system('tensorboard --logdir='+logdir)

def clear_tensorflow_log(logdir):
    os.system('rm ' + logdir +'/events.out.*')

# ***** GENERATING Simple DATA SETS for MACHINE LEARNING *****

# Generate all bit vectors of a given length (num_bits).
def gen_all_bit_vectors(num_bits):
    def bits(n):
        s = bin(n)[2:]
        return [int(b) for b in '0'*(num_bits - len(s))+s]
    return [bits(i) for i in range(2**num_bits)]

# Convert an integer to a bit vector of length num_bits, with prefix 0's as padding when necessary.
def int_to_bits(i,num_bits):
    s = bin(i)[2:]
    return [int(b) for b in '0' * (num_bits - len(s)) + s]

def all_ints_to_bits(num_bits):
    return [int_to_bits(i) for i in range(2**num_bits)]

# Convert an integer k to a sparse vector in which all bits are "off" except the kth bit.

def int_to_one_hot(int,size,off_val=0, on_val=1,floats=False):
    if floats:
        off_val = float(off_val); on_val = float(on_val)
    if int < size:
        v = [off_val] * size
        v[int] = on_val
        return v

# Generate all one-hot vectors of length len
def all_one_hots(len, floats=False):
    return [int_to_one_hot(i,len,floats=floats) for i in range(len)]

# ****** RANDOM VECTORS of Chosen Density *****

# Given a density (fraction), this randomly places onvals to produce a vector with the desired density.
def gen_dense_vector(size, density=.5, onval=1, offval=0):
    a = [offval] * size
    indices = np.random.choice(size,round(density*size),replace=False)
    for i in indices: a[i] = onval
    return a

def gen_random_density_vectors(count,size,density_range=(0,1)):
    return [gen_dense_vector(size,density=np.random.uniform(*density_range)) for c in range(count)]

# ****** LINES (horiz and vert) in arrays *********

# This produces either rows or columns of values (e.g. 1's), where the bias controls whether or not
# the entire row/column gets filled in or not just some cells. bias=1 => fill all.  Indices are those of the
# rows/columns to fill.  This is mainly used for creating data sets for classification tasks: horiz -vs- vertical
# lines.

def gen_line_array(dims,indices,line_item=1,background=0,columns=False,bias=1.0):
    a = np.array([background]*np.prod(dims)).reshape(dims)
    if columns: a = a.reshape(list(reversed(dims)))
    for row in indices:
        for j in range(a.shape[1]):
            if np.random.uniform(0, 1) <= bias: a[row,j] = line_item
    if columns: a = a.transpose()
    return a


# ****** ML CASE GENERATORS *****
# A ML "case" is a vector with two elements: the input vector and the output (target) vector.  These are the
# high-level functions that should get called from ML code.  They invoke the supporting functions above.

# The simplest autoencoders use the set of one-hot vectors as inputs and target outputs.

def gen_all_one_hot_cases(len, floats=False):
    return [[c,c] for c in all_one_hots(len,floats=floats)]

# Produce a list of pairs, with each pair consisting of a num_bits bit pattern and a singleton list containing
# the parity bit: 0 => an even number of 1's, 1 => odd number of 1's.  When double=True, a 2-bit vector is the
# target, with bit 0 indicating even parity and bit 1 indicating odd parity.

def gen_all_parity_cases(num_bits, double=True):
    def parity(v): return sum(v) % 2
    def target(v):
        if double:
            tg = [0,0].copy()
            tg[parity(v)] = 1
            return tg
        else: return [parity(v)]

    return [[c, target(c)] for c in gen_all_bit_vectors(num_bits)]

# This produces "count" cases, where features = random bit vectors and target = a one-hot vector indicating
# the number of 1's in the feature vector.  Note that the target vector is one bit larger than the feature
# vector to account for the case of a zero-sum feature vector.

def gen_vector_count_cases(num,size,drange=(0,1)):
    feature_vectors = gen_random_density_vectors(num,size,density_range=drange)
    target_vectors = [int_to_one_hot(sum(fv),size+1) for fv in feature_vectors]
    return [[fv,tv] for fv,tv in zip(feature_vectors,target_vectors)]

# Generate cases whose feature vectors, when converted into 2-d arrays, contain either a horizontal or
# a vertical line.  The class is then simply horizontal or vertical.

def gen_random_line_cases(num_cases,dims,min_lines=1,min_opens=1,bias=1.0, classify=True,
                          line_item=1,background=0,flat=True,floats=False):
    def gen_features(r_or_c):
        r_or_c = int(r_or_c)
        size = dims[r_or_c]
        count = np.random.randint(min_lines,size-min_opens+1)
        return gen_line_array(dims,indices=np.random.choice(size,count,replace=False), line_item=line_item,
                              background=background,bias=bias,columns =(r_or_c == 1))
    def gen_case():
        label = np.random.choice([0,1]) # Randomly choose to use a row or a column
        if classify:  # It's a classification task, so use 2 neurons, one for each class (horiz, or vert)
            target = [0]*2
            target[label] = 1
        else: target = [float(label)]
        features = gen_features(label)
        if flat: features = features.flatten().tolist()
        return (features, target)

    if floats:
        line_item = float(line_item); background = float(background)
    return [gen_case() for i in range(num_cases)]

# ***** PRIMITIVE DATA VIEWING ******

def show_results(grabbed_vals,grabbed_vars=None,dir='probeview'):
    showvars(grabbed_vals,names = [x.name for x in grabbed_vars], msg="The Grabbed Variables:")

def showvars(vals,names=None,msg=""):
    print("\n"+msg,end="\n")
    for i,v in enumerate(vals):
        if names: print("   " + names[i] + " = ",end="\n")
        print(v,end="\n\n")

# *******  DATA PLOTTING ROUTINES *********

def simple_plot(yvals,xvals=None,xtitle='X',ytitle='Y',title='Y = F(X)'):
    xvals = xvals if xvals is not None else list(range(len(yvals)))
    PLT.plot(xvals,yvals)
    PLT.xlabel(xtitle); PLT.ylabel(ytitle); PLT.title(title)
    PLT.draw()

# Each history is a list of pairs (timestamp, value).
def plot_training_history(error_hist,validation_hist=[],xtitle="Epoch",ytitle="Error",title="History",fig=True):
    PLT.ion()
    if fig: PLT.figure()
    if len(error_hist) > 0:
        simple_plot([p[1] for p in error_hist], [p[0] for p in error_hist],xtitle=xtitle,ytitle=ytitle,title=title)
        PLT.hold(True)
    if len(validation_hist) > 0:
        simple_plot([p[1] for p in validation_hist], [p[0] for p in validation_hist])
    PLT.ioff()

# alpha = transparency
def simple_scatter_plot(points,alpha=0.5,radius=3):
    colors = ['red','green','blue','magenta','brown','yellow','orange','brown','purple','black']
    a = np.array(points).transpose()
    PLT.scatter(a[0],a[1],c=colors,alpha=alpha,s=np.pi*radius**2)
    PLT.draw()

# This is Hinton's classic plot of a matrix (which may represent snapshots of weights or a time series of
# activation values.  Each value is represented by a red (positive) or blue (negative) square whose size reflects
# the absolute value.  This works best when maxsize is hardwired to 1.

def hinton_plot(matrix, maxval=None, maxsize=1, fig=None,trans=False,scale=True, title='Hinton plot'):
    hfig = fig if fig else PLT.figure()
    hfig.suptitle(title,fontsize=18)
    if trans: matrix = matrix.transpose()
    if maxval == None: maxval = np.abs(matrix).max()
    if not maxsize: maxsize = 2**np.ceil(np.log(maxval)/np.log(2))

    axes = hfig.gca()
    axes.clear()
    axes.patch.set_facecolor('gray');  # This is the background color.  Hinton uses gray
    axes.set_aspect('auto','box')  # Options: ('equal'), ('equal','box'), ('auto'), ('auto','box')..see matplotlib docs
    axes.xaxis.set_major_locator(PLT.NullLocator()); axes.yaxis.set_major_locator(PLT.NullLocator())

    for (x, y), val in np.ndenumerate(matrix):
        color = 'red' if val > 0 else 'blue'  # Hinton uses white = pos, black = neg
        if scale: size = np.sqrt(min(maxsize,maxsize*np.abs(val)/maxval))
        else: size = np.sqrt(min(np.abs(val),maxsize))  # The original version did not include scaling
        bottom_left = [x - size / 2, y - size / 2]
        blob = PLT.Rectangle(bottom_left, size, size, facecolor=color, edgecolor='white')
        axes.add_patch(blob)
    axes.autoscale_view()
    axes.invert_yaxis()  # Draw row zero on TOP
    PLT.draw()
    PLT.pause(1)
    axes.invert_yaxis()  # Re-invert, else next drawing on same axis will be upside-down.

# ****** Principle Component Analysis (PCA) ********
# This performs the basic operations outlined in "Python Machine Learning" (pp.128-135).  It begins with
# an N x K array whose rows are cases and columns are features.  It then computes the covariance matrix (of features),
# which is then used to compute the eigenvalues and eigenvectors.  The eigenvectors corresponding to the largest
# (absolute value) eigenvalues are then combined to produce a transformation matrix, which is applied to the original
# N cases to produce N new cases, each with J (ideally J << K) features.  This is UNSUPERVISED dimension reduction.

def pca(features,target_size,bias=True,rowvar=False):
    farray = features if isinstance(features,np.ndarray) else np.array(features)
    cov_mat = np.cov(farray,rowvar=rowvar,bias=bias) # rowvar=False => each var's values are in a COLUMN.
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    return gen_dim_reduced_data(farray,target_size,eigen_vals, eigen_vecs)

# Use the highest magnitude eigenvalues (and their eigenvectors) as the basis for feature-vector transformations that
# reduce the dimensionality of the data.  feature_array is N x M, where N = # cases, M = # features

def gen_dim_reduced_data(feature_array,target_size,eigen_values,eigen_vectors):
    eigen_pairs = [(np.abs(eigen_values[i]),eigen_vectors[:,i]) for i in range(len(eigen_values))]
    eigen_pairs.sort(key=(lambda p: p[0]),reverse=True)  # Sorts tuples by their first element = abs(eigenvalue)
    best_vectors = [pair[1] for pair in eigen_pairs[ : target_size]]
    w_transform = np.array(best_vectors).transpose()
    return np.dot(feature_array,w_transform)
