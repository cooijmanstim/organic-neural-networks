import os
import gzip
import cPickle

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

    which_sets = "train valid test".split()
    return dict((which_set, dict(features=x.astype("float32"),
                                 targets=y.astype("int32")))
                for which_set, (x, y) in zip(which_sets, split))
