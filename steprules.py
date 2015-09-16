import theano
import theano.tensor as T

def scale(scale=1):
    return lambda parameter, step: scale * step

def rmsprop(scale=1, decay_rate=0.9, max_scaling=1e5):
    def compute(parameter, step):
        prev_mss = theano.shared(value=parameter.get_value() * 0)
        mss = ((1 - decay_rate) * T.sqr(step)
               + decay_rate * prev_mss)
        step *= scale / T.maximum(T.sqrt(mss), 1. / max_scaling)
        updates = [(prev_mss, mss)]
        return step, updates
    return compute
