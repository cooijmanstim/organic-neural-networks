import sys
from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T
import util, activation, initialization, steprules, whitening, mnist

learning_rate = 1e-3
# use batch normalization in addition to PRONG (i.e. PRONG+)
batch_normalize = False

data = mnist.get_data()
n_outputs = 10

dims = [784, 500, 300, 100, n_outputs]
layers = [
    dict(f=activation.tanh,
         c=util.shared_floatx((m,),   initialization.constant(0)),   # input mean
         U=util.shared_floatx((m, m), initialization.identity()),    # input whitening matrix
         W=util.shared_floatx((m, n), initialization.orthogonal()),  # weight matrix
         g=util.shared_floatx((n,),   initialization.constant(1)),   # gammas (for batch normalization)
         b=util.shared_floatx((n,),   initialization.constant(0)))   # bias
    for m, n in util.safezip(dims[:-1], dims[1:])]
layers[-1]["f"] = activation.logsoftmax

features, targets = T.matrix("features"), T.ivector("targets")

#theano.config.compute_test_value = "warn"
#features.tag.test_value = data["valid"]["features"][:11]
#targets.tag.test_value = data["valid"]["targets"][:11]

# reparametrization updates
reparameterization_updates = []
# theano graphs with assertions & breakpoints, to be evaluated after
# performing the updates
reparameterization_checks = []

# construct theano graph
h = features
for i, layer in enumerate(layers):
    f, c, U, W, g, b = [layer[k] for k in "fcUWgb"]

    # construct reparameterization graph
    updates, checks = whitening.get_updates(
        h, c, U, V=W, d=b,
        decomposition="svd", zca=True, bias=1e-3)
    reparameterization_updates.extend(updates)
    reparameterization_checks.extend(checks)

    # whiten input
    h = T.dot(h - c, U)
    # compute layer as usual
    h = T.dot(h, W)
    if batch_normalize:
        h -= h.mean(axis=0, keepdims=True)
        h /= T.sqrt(h.var(axis=0, keepdims=True) + 1e-8)
        h *= g
    h += b
    h = f(h)

logp = h
cross_entropy = -logp[T.arange(logp.shape[0]), targets]
cost = cross_entropy.mean(axis=0)

parameters = [layer[k] for k in ("cUWgb" if batch_normalize else "cUWb") for layer in layers]
gradients = OrderedDict(zip(parameters, T.grad(cost, parameters)))
steps = [(parameter, parameter - learning_rate * gradient)
         for parameter, gradient in gradients.items()]

# compile theano functions
monitor_fn = theano.function([features, targets], cost)
step_fn = theano.function([features, targets], updates=steps)
reparameterization_fn = theano.function([features], updates=reparameterization_updates)
check_fn = theano.function([features], reparameterization_checks)

# train for a couple of epochs
nepochs = 10
batch_size = 100
# reparameterize every 100 steps
interval = 100
for i in range(nepochs):
    print(i, "train cross entropy", monitor_fn(**data["train"]))
    print(i, "training")
    for j, a in enumerate(range(0, len(data["train"]["features"]), batch_size)):
        if j % interval == 0:
            # whiten on a subset of data
            reparameterization_fn(data["train"]["features"][:1000])
            # perform sanity checks
            # NOTE: these WILL fail regularly due to numerical issues, e.g. if input is not full rank
            #check_fn(data["train"]["features"][:1000])

        b = a + batch_size
        step_fn(**util.slice_sources(data["train"], a, b))
        sys.stdout.write(".")
        sys.stdout.flush()
    print()
    print(i, "done")
