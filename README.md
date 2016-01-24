# organic-neural-networks
Experiments with Natural Neural Networks

`prong.py` contains a Theano implementation of the PRONG and PRONG+ algorithms from the paper Natural Neural Networks by Desjardins et al. (http://arxiv.org/abs/1507.00210). I run it like this:

```THEANO_FLAGS="floatX=float32" MNIST_PKL_GZ=~/datasets/mnist.pkl.gz python prong.py```

If `MNIST_PKL_GZ` does not exist, the dataset will be downloaded and placed in that location.
