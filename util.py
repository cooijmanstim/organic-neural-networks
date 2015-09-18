import itertools
import numpy as np
import theano
import theano.tensor as T

rng = np.random.RandomState(12345)

def safezip(*iterables):
    xss = list(map(list, iterables))
    if any(len(xs) != len(xss[0]) for xs in xss[1:]):
        raise ValueError("zipping iterables of unequal length")
    return zip(*xss)

def interleave(*iterables):
    return itertools.chain.from_iterable(zip(*iterables))

def shared_floatx(shape, initializer):
    return theano.shared(initializer(rng, shape).astype(theano.config.floatX))

def batched_flatcat(xs):
    return T.concatenate([x.flatten(ndim=2) for x in xs],
                         axis=1)

def matsave(path, arr):
    from PIL import Image
    import matplotlib.cm
    import scipy.misc
    im = Image.fromarray(matplotlib.cm.jet(arr, bytes=True))
    scipy.misc.imsave(path, im)
