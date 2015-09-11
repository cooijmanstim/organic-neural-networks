import itertools
import numpy as np
import theano

rng = np.random.RandomState(0)

def interleave(*iterables):
    return itertools.chain.from_iterable(zip(*iterables))

def shared_floatx(shape, initializer):
    return theano.shared(initializer.generate(rng, shape).astype(theano.config.floatX))

