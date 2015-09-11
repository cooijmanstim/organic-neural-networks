import theano.tensor as T

identity = lambda x: x
tanh = T.tanh
rectifier = lambda x: x * (x > 0)
softmax = T.nnet.softmax
def logsoftmax(x):
    # take out common factor of numerator/denominator for stability
    x -= x.max(axis=1, keepdims=True)
    return x - T.log(T.exp(x).sum(axis=1, keepdims=True))

