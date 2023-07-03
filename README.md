# deno-ann
A simple artifical neural network library I wrote to study deep learning.

This is a library of routines written in Typescript for buliding and training artificial neural networks, loosely based on the code samples in [Deep Learning from Scratch by Seth Weidman](https://learning.oreilly.com/library/view/deep-learning-from/9781492041405/).

The neural network routines here are not based on automatic differentiation. The codes to propagate gradients backward have to be written manually. Writing backward pass manually is a tidious process.

Also only the simplest models can be trained in a reasonable time because the numerical routines are not optimized.

This libray is what [Toy - MNIST Classification](https://jaeyoung.se/toys/mnist) and [Toy - Generate text with character-based RNN](https://jaeyoung.se/toys/rnn) are based on.

## Directories

* `neural/` - contains the basic building blocks that can be used to build and train artificial neural networks.
* `examples/mnist/` - MNIST classification example.
* `ndarray/` - multi-dimensional arrays a la numpy
* `trees/` - functions to manipulate tree-like structures a la [Pytrees](https://jax.readthedocs.io/en/latest/pytrees.html)
* `arrays/` - array utilities
* `floats/` - Float32Array utilities
* `numerical/` - mathematical utilities
