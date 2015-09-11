import numpy as np

import theano
import theano.tensor as T
from theano.tests.breakpoint import PdbBreakpoint

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

def compute_parameters(x, bias=1e-5):
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

def get_updates(h, c, U, V, d, bias=1e-5):
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
    newc, newU = compute_parameters(h, bias)
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
