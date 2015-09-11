import numpy as np

import theano
import theano.tensor as T

from blocks import initialization

import util
import activation
import whitening
import mnist

[(train_x, train_y), (valid_x, valid_y), (test_x, test_y)] = mnist.get_data()

features = T.matrix("features")
targets = T.ivector("targets")

theano.config.compute_test_value = "warn"
features.tag.test_value = valid_x
targets.tag.test_value = valid_y

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

dims = [49, 10, 10, 10]
fs = [activation.rectifier for _ in dims[1:-1]] + [activation.logsoftmax]
if whiten_inputs:
    cs = [util.shared_floatx((m,), initialization.Constant(0))
          for m in dims[:-1]]
    Us = [util.shared_floatx((m, m), initialization.Identity())
          for m in dims[:-1]]
Ws = [util.shared_floatx((m, n), initialization.Orthogonal())
      for m, n in zip(dims, dims[1:])]
if batch_normalize:
    gammas = [util.shared_floatx((n, ), initialization.Constant(1))
              for n in dims[1:]]
bs = [util.shared_floatx((n, ), initialization.Constant(0))
      for n in dims[1:]]

updates = []
# theano graphs with assertions & breakpoints, to be evaluated after
# performing the updates
checks = []

h = x
for i, (W, b, f) in enumerate(zip(Ws, bs, fs)):
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
cross_entropy = yhat[T.arange(yhat.shape[0]), targets].mean(axis=0)

parameters = [Ws, gammas, bs] if batch_normalize else [Ws, bs]
gradients = T.grad(cross_entropy, list(util.interleave(*parameters)))

flat_gradient = T.concatenate(
    [gradient.ravel() for gradient in gradients],
    axis=0)

fisher = (flat_gradient.dimshuffle(0, "x") *
          flat_gradient.dimshuffle("x", 0))

# run updates & checks
theano.function([features], updates=updates)(train_x)
theano.function([features], checks)(train_x)

np_fisher, np_gradient = theano.function([features, targets], [fisher, flat_gradient])(train_x, train_y)

import matplotlib.pyplot as plt

plt.figure()
plt.hist(np_gradient, bins=30)
plt.title("gradient histogram")

plt.figure()
plt.hist(np_fisher.ravel(), bins=30)
plt.title("fisher histogram")

plt.matshow(abs(np_fisher))

plt.show()

import scipy.misc

scipy.misc.imsave("F.png", np_fisher)

print "condition number:"
print np.linalg.cond(np_fisher)
