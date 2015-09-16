import numpy as np

def constant(x):
    return lambda rng, shape: x * np.ones(shape)

def identity():
    return lambda rng, (m, n): np.eye(m, n)

def scaled(scale, initializer):
    return lambda rng, shape: scale * initializer(rng, shape)

def isotropic_gaussian(mean=0, std=1):
    return lambda rng, shape: rng.normal(mean, std, size=shape)

def orthogonal(scale=1.1):
    # taken from https://gist.github.com/kastnerkyle/f7464d98fe8ca14f2a1a
    def generate(rng, shape):
        """ benanne lasagne ortho init (faster than qr approach)"""
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = rng.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return scale * q[:shape[0], :shape[1]]
    return generate
