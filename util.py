import itertools
import numpy as np
import theano

rng = np.random.RandomState(0)

def safezip(*iterables):
    xss = list(map(list, iterables))
    if any(len(xs) != len(xss[0]) for xs in xss[1:]):
        raise ValueError("zipping iterables of unequal length")
    return zip(*xss)

def interleave(*iterables):
    return itertools.chain.from_iterable(zip(*iterables))

def shared_floatx(shape, initializer):
    return theano.shared(initializer(rng, shape).astype(theano.config.floatX))

