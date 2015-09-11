import os
import gzip
import cPickle

# adapted from https://gist.github.com/kastnerkyle/f7464d98fe8ca14f2a1a
def get_data(datasets_dir='/home/tim/datasets/mnist'):
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    f = gzip.open(data_file, 'rb')
    try:
        split = cPickle.load(f, encoding="latin1")
    except TypeError:
        split = cPickle.load(f)
    f.close()

    return [(x.astype("float32"), y.astype("int32"))
            for x, y in split]

