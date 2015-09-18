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

#theano.config.compute_test_value = "warn"
#features.tag.test_value = datasets["valid"]["features"]
#targets.tag.test_value = datasets["valid"]["targets"]

n_outputs = 10

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

dims = [49, 16, 16, 16, n_outputs]
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

# "eigh" on covariance or "svd" on data matrix
whitening_strategy = "svd"
# whether to use ZCA (i.e. preserve input representation as much as possible)
zca = True
# compute fisher based on supervised "loss" or model "output"
objective = "output"
# eigenvalue bias
bias = 1e-1

h = x
for i, (W, b, f) in enumerate(util.safezip(Ws, bs, fs)):
    if whiten_inputs:
        c, U = cs[i], Us[i]
        wupdates, wchecks = whitening.get_updates(h, c, U, V=W, d=b,
                                                  strategy=whitening_strategy,
                                                  zca=zca, bias=bias)
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

logp = h
cross_entropy = -logp[T.arange(logp.shape[0]), targets]
cost = cross_entropy.mean(axis=0)

parameters = list(util.interleave(*([Ws, gammas, bs] if batch_normalize else [Ws, bs])))
n_parameters_per_layer = 3 if batch_normalize else 2
gradients = OrderedDict(zip(parameters, T.grad(cost, parameters)))

hidden_parameters = parameters
hidden_parameters = hidden_parameters[n_parameters_per_layer:]  # remove visible layer parameters
#hidden_parameters = hidden_parameters[:-n_parameters_per_layer] # remove softmax layer parameters

def estimate_fisher(outputs, n_outputs, parameters):
    # shape (sample_size, n_outputs, #parameters)
    grads = T.stack(*[util.batched_flatcat(
        T.jacobian(outputs[:, j],
                   hidden_parameters))
        for j in xrange(n_outputs)])
    # ravel the batch and output axes so that the product will sum
    # over the outputs *and* over the batch. divide by the batch
    # size to get the batch mean.
    grads = grads.reshape((grads.shape[0] * grads.shape[1], grads.shape[2]))
    fisher = T.dot(grads.T, grads) / grads.shape[0]
    return fisher

# estimate_fisher will perform one backpropagation per sample, so
# don't go wild
sample_size = 100
if objective == "loss":
    # fisher on loss
    fisher = estimate_fisher(cross_entropy[:sample_size, np.newaxis],
                             1, hidden_parameters)
elif objective == "output":
    # fisher on output
    fisher = estimate_fisher(logp[:sample_size, :],
                             n_outputs, hidden_parameters)

steps = []
step_updates = []
for parameter, gradient in gradients.items():
    step, steprule_updates = steprule(parameter, gradient)
    steps.append((parameter, -step))
    step_updates.append((parameter, parameter - step))
    step_updates.extend(steprule_updates)

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

np_fisher_before = compute(fisher, which_set="train")
compute(updates=updates, which_set="train")
compute(checks, which_set="train")
np_fisher_after = compute(fisher, which_set="train")

prefix = "F%s_%s" % (objective, whitening_strategy)
util.matsave("%s_before.png" % prefix, abs(np_fisher_before))
util.matsave("%s_after.png" % prefix, abs(np_fisher_after))

nepochs = 0
batch_size = 100
for i in xrange(nepochs):
    print i, "train cross entropy", compute(cost, which_set="train")
    compute(updates=updates, which_set="train")
    compute(checks, which_set="train")
    np_fisher = compute(fisher, which_set="train")
    util.matsave("%s_%i.png" % (prefix, i), abs(np_fisher))
    print i, "training"
    for a in xrange(0, len(datasets["train"]["features"]), batch_size):
        b = a + batch_size
        compute(updates=step_updates, which_set="train", subset=slice(a, b))
        sys.stdout.write(".")
        sys.stdout.flush()
    print
    print i, "done"

print nepochs, "train cross entropy", compute(cost, which_set="train")
