import sys
import collections
from collections import OrderedDict

import numpy as np

import theano
import theano.tensor as T

import util
import activation
import initialization
import steprules
import whitening
import mnist

datasets = mnist.get_data()

features = T.matrix("features")
targets = T.ivector("targets")

theano.config.compute_test_value = "warn"
features.tag.test_value = datasets["valid"]["features"]
targets.tag.test_value = datasets["valid"]["targets"]

# downsample to keep number of parameters low
x = features
x = x.reshape((x.shape[0], 1, 28, 28))
reduction = 4
x = T.nnet.conv.conv2d(
    x,
    (np.ones((1, 1, reduction, reduction),
             dtype=np.float32)
     / reduction**2),
    subsample=(reduction, reduction))
x = x.flatten(ndim=2)

batch_normalize = False
whiten_inputs = True
steprule = steprules.rmsprop(scale=1e-3)

dims = [49, 10, 10, 10]
fs = [activation.tanh for _ in dims[1:-1]] + [activation.logsoftmax]
if whiten_inputs:
    cs = [util.shared_floatx((m,), initialization.constant(0))
          for m in dims[:-1]]
    Us = [util.shared_floatx((m, m), initialization.identity())
          for m in dims[:-1]]
Ws = [util.shared_floatx((m, n), initialization.orthogonal())
      for m, n in util.safezip(dims[:-1], dims[1:])]
if batch_normalize:
    gammas = [util.shared_floatx((n, ), initialization.constant(1))
              for n in dims[1:]]
bs = [util.shared_floatx((n, ), initialization.constant(0))
      for n in dims[1:]]

updates = []
# theano graphs with assertions & breakpoints, to be evaluated after
# performing the updates
checks = []

h = x
for i, (W, b, f) in enumerate(util.safezip(Ws, bs, fs)):
    if whiten_inputs:
        c, U = cs[i], Us[i]
        wupdates, wchecks = whitening.get_updates(h, c, U, V=W, d=b)
        updates.extend(wupdates)
        checks.extend(wchecks)
        h = T.dot(h - c, U)

    h = T.dot(h, W)

    if batch_normalize:
        mean = h.mean(axis=0, keepdims=True)
        var  = h.var (axis=0, keepdims=True)
        h = (h - mean) / T.sqrt(var + 1e-16)
        h *= gammas[i]

    h += b
    h = f(h)

yhat = h
cross_entropy = -yhat[T.arange(yhat.shape[0]), targets].mean(axis=0)

parameters = list(util.interleave(*([Ws, gammas, bs] if batch_normalize else [Ws, bs])))
gradients = OrderedDict(zip(parameters, T.grad(cross_entropy, parameters)))

steps = []
step_updates = []
for parameter, gradient in gradients.items():
    step, steprule_updates = steprule(parameter, gradient)
    steps.append((parameter, -step))
    step_updates.append((parameter, parameter - step))
    step_updates.extend(steprule_updates)

flat_gradient = T.concatenate(
    [gradient.ravel() for gradient in gradients.values()],
    axis=0)

fisher = (flat_gradient.dimshuffle(0, "x") *
          flat_gradient.dimshuffle("x", 0))

def plot(epoch, fisher):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(fisher.ravel(), bins=30)
    plt.title("fisher histogram")

    plt.matshow(abs(fisher))

    plt.show()

def dump(epoch, fisher):
    np.savez("dump/fisher_%i.npz", fisher)

    import scipy.misc

    scipy.misc.imsave("F_%i.png" % epoch, fisher)

#   print epoch, "condition number:"
#   print np.linalg.cond(fisher)

# super pythonic yo
def tupelo(x):
    try:
        return tuple(x)
    except TypeError:
        return x

compile_memo = dict()
def compile(variables=(), updates=()):
    key = (tupelo(variables),
           tuple(OrderedDict(updates).items()))
    try:
        return compile_memo[key]
    except KeyError:
        return compile_memo.setdefault(
            key,
            theano.function(
                [features, targets],
                variables,
                updates=updates,
                on_unused_input="ignore"))

# compile theano function and compute on select data
def compute(variables=(), updates=(), which_set=None, subset=None):
    return (
        compile(variables=variables, updates=updates)(
            **datadict(which_set, subset)))

def datadict(which_set, subset=None):
    dataset = datasets[which_set]
    return dict(
        (source,
         dataset[source]
         if subset is None
         else dataset[source][subset])
        for source in "features targets".split())

nepochs = 3
batch_size = 100
for i in xrange(nepochs):
    print i, "train cross entropy", compute(cross_entropy, which_set="train")
    compute(updates=updates, which_set="train")
    compute(checks, which_set="train")
    np_fisher = compute(fisher, which_set="train")
    plot(i, np_fisher)
    dump(i, np_fisher)
    print i, "training"
    for a in xrange(0, len(datasets["train"]["features"]), batch_size):
        b = a + batch_size
        compute(updates=step_updates, which_set="train", subset=slice(a, b))
        sys.stdout.write(".")
        sys.stdout.flush()
    print
    print i, "done"

print nepochs, "train cross entropy", compute(cross_entropy, which_set="train")
