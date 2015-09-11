import os
import gzip
import cPickle
import itertools

import numpy as np

import theano
import theano.tensor as T
from theano.tests.breakpoint import PdbBreakpoint

import matplotlib.pyplot as plt

from blocks import initialization

# compute and show fisher matrix for all combinations of initialization, nonlinearity and layerwise normalization

rng = np.random.RandomState(0)

def interleave(*iterables):
    return itertools.chain.from_iterable(zip(*iterables))

def shared_floatx(shape, initializer):
    return theano.shared(initializer.generate(rng, shape).astype(theano.config.floatX))

identity = lambda x: x
tanh = T.tanh
rectifier = lambda x: x * (x > 0)
softmax = T.nnet.softmax
def logsoftmax(x):
    # take out common factor of numerator/denominator for stability
    x -= x.max(axis=1, keepdims=True)
    return x - T.log(T.exp(x).sum(axis=1, keepdims=True))

# adapted from https://gist.github.com/kastnerkyle/f7464d98fe8ca14f2a1a
def mnist(datasets_dir='/home/tim/datasets/mnist'):
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    f = gzip.open(data_file, 'rb')
    try:
        split = cPickle.load(f, encoding="latin1")
    except TypeError:
        split = cPickle.load(f)
    f.close()

    return [(x.astype("float32"), y.astype("int32"))
            for x, y in split]

[(train_x, train_y), (valid_x, valid_y), (test_x, test_y)] = mnist()

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
fs = [rectifier for _ in dims[1:-1]] + [logsoftmax]
if whiten_inputs:
    cs = [shared_floatx((m,), initialization.Constant(0))
          for m in dims[:-1]]
    Us = [shared_floatx((m, m), initialization.Identity())
          for m in dims[:-1]]
Ws = [shared_floatx((m, n), initialization.Orthogonal())
      for m, n in zip(dims, dims[1:])]
if batch_normalize:
    gammas = [shared_floatx((n, ), initialization.Constant(1))
              for n in dims[1:]]
bs = [shared_floatx((n, ), initialization.Constant(0))
      for n in dims[1:]]


# from theano.tensor.nlinalg but float32
numpy = np
class lstsq(theano.gof.Op):
    __props__ = ()

    def make_node(self, x, y, rcond):
        x = theano.tensor.as_tensor_variable(x)
        y = theano.tensor.as_tensor_variable(y)
        rcond = theano.tensor.as_tensor_variable(rcond)
        return theano.Apply(self, [x, y, rcond],
                            [theano.tensor.fmatrix(), theano.tensor.fvector(),
                             theano.tensor.lscalar(), theano.tensor.fvector()])

    def perform(self, node, inputs, outputs):
        zz = numpy.linalg.lstsq(inputs[0], inputs[1], inputs[2])
        outputs[0][0] = zz[0]
        outputs[1][0] = zz[1]
        outputs[2][0] = numpy.array(zz[2])
        outputs[3][0] = zz[3]


def compute_whitening_parameters(x, bias=1e-5):
    n = x.shape[0].astype(theano.config.floatX)
    c = x.mean(axis=0)
    # we can call svd on the covariance rather than the data, but that
    # seems to lose accuracy
    _, s, vT = T.nlinalg.svd(x - c, full_matrices=False)
    # the covariance will be I / (n - 1); introduce a factor
    # sqrt(n - 1) here to compensate
    d = T.sqrt(n - 1) / (s + bias)
    U = T.dot(vT.T * d, vT)
    return c, U

def get_whitening_updates(h, c, U, V, d, bias=1e-5):
    updates = []
    checks = []

    # theano applies updates in parallel, so all updates are in terms
    # of the old values.  use this and assign the return value, i.e.
    # x = update(x, foo()).  x is then a non-shared variable that
    # refers to the updated value.
    def update(variable, new_value):
        updates.append((variable, new_value))
        return new_value

    # compute canonical parameters
    W = T.dot(U, V)
    b = d - T.dot(c, W)

    # update estimates of c, U
    newc, newU = compute_whitening_parameters(h, bias)
    c = update(c, newc)
    U = update(U, newU)

    # check that the new covariance is indeed identity
    n = h.shape[0].astype(theano.config.floatX)
    covar = T.dot(h.T, h) / (n - 1)
    whiteh = T.dot(h - c, U)
    whitecovar = T.dot(whiteh.T, whiteh) / (n - 1)
    checks.append(PdbBreakpoint
                  ("correlated after whitening")
                  (1 - T.allclose(whitecovar,
                                  T.identity_like(whitecovar),
                                  rtol=1e-3, atol=1e-3),
                   c, U, covar, whitecovar, h)[0])

    # adjust V, d so that the total transformation is unchanged
    # (lstsq is much more stable than T.inv)
    V = update(V, lstsq()(U, W, -1)[0])
    d = update(d, b + T.nlinalg.matrix_dot(c, U, V))

    # check that the total transformation is unchanged
    before = b + T.dot(h, W)
    after = d + T.nlinalg.matrix_dot(h - c, U, V)
    checks.append(
        PdbBreakpoint
        ("transformation changed")
        (1 - T.allclose(before, after,
                        rtol=1e-3, atol=1e-3),
         T.constant(0.0), W, b, c, U, V, d, h, before, after)[0])

    return updates, checks

updates = []
# theano graphs with assertions & breakpoints, to be evaluated after
# performing the updates
checks = []

h = x
for i, (W, b, f) in enumerate(zip(Ws, bs, fs)):
    if whiten_inputs:
        c, U = cs[i], Us[i]
        wupdates, wchecks = get_whitening_updates(h, c, U, V=W, d=b)
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
gradients = T.grad(cross_entropy, list(interleave(*parameters)))

flat_gradient = T.concatenate(
    [gradient.ravel() for gradient in gradients],
    axis=0)

fisher = (flat_gradient.dimshuffle(0, "x") *
          flat_gradient.dimshuffle("x", 0))

# run updates & checks
theano.function([features], updates=updates)(train_x)
theano.function([features], checks)(train_x)

np_fisher, np_gradient = theano.function([features, targets], [fisher, flat_gradient])(train_x, train_y)

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
