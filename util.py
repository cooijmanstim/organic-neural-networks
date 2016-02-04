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
    # normalize
    arr -= arr.min()
    arr /= arr.max()
    # rgb using matplotlib's default colormap
    im = Image.fromarray(matplotlib.cm.jet(arr, bytes=True))
    scipy.misc.imsave(path, im)

# super pythonic yo
def tupelo(x):
    try:
        return tuple(x)
    except TypeError:
        return x

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

def slice_sources(dataset, *slice_args):
    s = slice(*slice_args)
    return dict((source_name, source[s])
                for source_name, source in dataset.items())
