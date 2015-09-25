import sys
from collections import OrderedDict
import yaml
import itertools

import numpy as np

import theano
import theano.tensor as T

import util
import activation
import initialization
import steprules
import whitening
import mnist

n_outputs = 10
steprule = steprules.rmsprop(scale=1e-3)

hyperparameters = dict(
    # "eigh" on covariance matrix or "svd" on data matrix
    decomposition="svd",
    # whether to remix after whitening
    zca=True,
    # compute fisher based on supervised "loss" or model "output"
    objective="output",
    # eigenvalue bias
    bias=1e-3,
    batch_normalize=True,
    whiten_inputs=False,
    share_parameters=True)

datasets = mnist.get_data()

features = T.matrix("features")
targets = T.ivector("targets")

#theano.config.compute_test_value = "warn"
#features.tag.test_value = datasets["valid"]["features"]
#targets.tag.test_value = datasets["valid"]["targets"]


# compilation helpers
compile_memo = dict()
def compile(variables=(), updates=()):
    key = (util.tupelo(variables),
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


# downsample input to keep number of parameters low
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

# input, hidden and output dims
dims = [(28 / reduction)**2, 16, 16, 16, n_outputs]

# allocate parameters
fs = [activation.tanh for _ in dims[1:-1]] + [activation.logsoftmax]
# layer input means
cs = [util.shared_floatx((m,), initialization.constant(0))
        for m in dims[:-1]]
# layer input whitening matrices
Us = [util.shared_floatx((m, m), initialization.identity())
        for m in dims[:-1]]
# weight matrices
Ws = [util.shared_floatx((m, n), initialization.orthogonal())
      for m, n in util.safezip(dims[:-1], dims[1:])]
# batch normalization diagonal scales
gammas = [util.shared_floatx((n, ), initialization.constant(1))
            for n in dims[1:]]
# biases or betas
bs = [util.shared_floatx((n, ), initialization.constant(0))
      for n in dims[1:]]

# reparametrization updates
updates = []
# theano graphs with assertions & breakpoints, to be evaluated after
# performing the updates
checks = []

parameters_by_layer = []

# construct theano graph
h = x
for i in xrange(len(dims) - 1):
    layer_parameters = []

    if hyperparameters["share_parameters"]:
        # for the compatible layers
        if Ws[i].shape == Ws[1].shape:
            i = 1

    c, U, W, gamma, b, f = cs[i], Us[i], Ws[i], gammas[i], bs[i], fs[i]

    if hyperparameters["whiten_inputs"]:
        wupdates, wchecks = whitening.get_updates(
            h, c, U, V=W, d=b,
            decomposition=hyperparameters["decomposition"],
            zca=hyperparameters["zca"],
            bias=hyperparameters["bias"])
        updates.extend(wupdates)
        checks.extend(wchecks)
        h = T.dot(h - c, U)
        layer_parameters.extend([c, U])

    h = T.dot(h, W)
    layer_parameters.append(W)

    if hyperparameters["batch_normalize"]:
        mean = h.mean(axis=0, keepdims=True)
        var  = h.var (axis=0, keepdims=True)
        h = (h - mean) / T.sqrt(var + 1e-16)
        h *= gammas[i]
        layer_parameters.append(gammas[i])

    h += b
    layer_parameters.append(b)
    h = f(h)

    parameters_by_layer.append(layer_parameters)

if hyperparameters["share_parameters"]:
    # remove repeated parameters
    del parameters_by_layer[2:-1]

logp = h
cross_entropy = -logp[T.arange(logp.shape[0]), targets]
cost = cross_entropy.mean(axis=0)

def estimate_fisher(outputs, n_outputs, parameters):
    # shape (sample_size, n_outputs, #parameters)
    grads = T.stack(*[util.batched_flatcat(
        T.jacobian(outputs[:, j], parameters))
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
# don't include visible layer parameters; too many and not full rank
fisher_parameters = list(itertools.chain(*parameters_by_layer[1:]))
if hyperparameters["objective"] == "loss":
    # fisher on loss
    fisher = estimate_fisher(cross_entropy[:sample_size, np.newaxis],
                             1, fisher_parameters)
elif hyperparameters["objective"] == "output":
    # fisher on output
    fisher = estimate_fisher(logp[:sample_size, :],
                             n_outputs, fisher_parameters)

# maybe train for a couple of epochs and track the FIM
parameters = list(itertools.chain(*parameters_by_layer))
gradients = OrderedDict(zip(parameters, T.grad(cost, parameters)))
steps = []
step_updates = []
for parameter, gradient in gradients.items():
    step, steprule_updates = steprule(parameter, gradient)
    steps.append((parameter, -step))
    step_updates.append((parameter, parameter - step))
    step_updates.extend(steprule_updates)

# for these experiments, whiten once and then nevermore
compute(updates=updates, which_set="train")
compute(checks, which_set="train")

nepochs = 100
batch_size = 100
np_fishers = []
cross_entropies = []
for i in xrange(nepochs):
    np_fishers.append(compute(fisher, which_set="train"))
    cross_entropies.append(compute(cost, which_set="train"))
    print i, "train cross entropy", cross_entropies[-1]
    print i, "training"
    for a in xrange(0, len(datasets["train"]["features"]), batch_size):
        b = a + batch_size
        compute(updates=step_updates, which_set="train", subset=slice(a, b))
        sys.stdout.write(".")
        sys.stdout.flush()
    print
    print i, "done"
cross_entropies.append(compute(cost, which_set="train"))

identifier = hash(hyperparameters)
np.savez_compressed("fishers_%s.npz" % identifier,
                    fishers=np.asarray(np_fishers),
                    cross_entropies=np.asarray(cross_entropies))
cPickle.dump(hyperparameters,
             "hyperparameters_%s.yaml" % identifier)
