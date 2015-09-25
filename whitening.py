import theano
import theano.tensor as T
from theano.tests.breakpoint import PdbBreakpoint

import util

def whiten_by_svd(x, bias, zca):
    n = x.shape[0].astype(theano.config.floatX)
    _, s, vT = T.nlinalg.svd(x, full_matrices=False)
    # the covariance will be I / (n - 1); introduce a factor
    # sqrt(n - 1) here to compensate
    D = T.diag(T.sqrt((n - 1) / (s**2 + bias)))
    U = T.dot(D, vT)
    if zca:
        U = T.dot(vT.T, U)
    return U

def whiten_by_eigh(x, bias, zca):
    n = x.shape[0].astype(theano.config.floatX)
    covar = T.dot(x.T, x) / (n - 1)
    s, v = T.nlinalg.eigh(covar)
    D = T.diag(1.0 / T.sqrt(s**2 + bias))
    U = T.dot(D, v.T)
    if zca:
        U = T.dot(v, U)
    return U

whiten_by = dict(
    svd=whiten_by_svd,
    eigh=whiten_by_eigh)

def get_updates(h, c, U, V, d, bias=1e-5, decomposition="svd", zca=True):
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
    c = update(c, h.mean(axis=0))
    U = update(U, whiten_by[decomposition](h - c, bias, zca))

    # check that the new covariance is indeed identity
    n = h.shape[0].astype(theano.config.floatX)
    covar = T.dot((h - c).T, (h - c)) / (n - 1)
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
    V = update(V, util.lstsq()(U, W, -1)[0])
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
