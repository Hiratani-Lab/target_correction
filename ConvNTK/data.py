#
# Effective label shift and correction: MNIST-NTK Example
#
# Data generation
#
import sys
from itertools import permutations

import numpy as np
import jax.numpy as jnp
from jax import random

import tensorflow as tf
# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type='GPU')
import tensorflow_datasets as tfds

data_dir = '/tmp/tfds'


def one_hot(x, k, dtype=jnp.float32):
	"""Create a one-hot encoding of x of size k."""
	return jnp.array(x[:, None] == jnp.arange(k), dtype)


def whiten_data(xtmps):
	EPS = 1e-6
	return jnp.divide( xtmps - jnp.mean(xtmps, axis=0, keepdims=True), jnp.std(xtmps, axis=0, keepdims=True) + EPS )


def load_mnist_data():
	# Fetch full datasets for evaluation
	# tfds.load returns tf.Tensors (or tf.data.Datasets if batch_size != -1)
	# You can convert them to NumPy arrays (or iterables of NumPy arrays) with tfds.dataset_as_numpy
	mnist_data, info = tfds.load(name="mnist", batch_size=-1, data_dir=data_dir, with_info=True)
	mnist_data = tfds.as_numpy(mnist_data)
	train_data, test_data = mnist_data['train'], mnist_data['test']
	num_labels = info.features['label'].num_classes
	h, w, c = info.features['image'].shape
	num_pixels = h * w * c

	# Full train set
	train_images, train_labels = train_data['image'], train_data['label']
	#train_images = jnp.reshape(train_images, (len(train_images), num_pixels, 1))
	
	# Full test set
	test_images, test_labels = test_data['image'], test_data['label']
	#test_images = jnp.reshape(test_images, (len(test_images), num_pixels))
	
	#normalization
	#Z = 256.0 
	#train_images = (1/Z)*train_images
	#test_images = (1/Z)*test_images
	
    # whitening
	train_images = whiten_data( train_images )
	test_images = whiten_data( test_images )
	
	#one-hot label
	train_labels = one_hot(train_labels, num_labels)
	test_labels = one_hot(test_labels, num_labels)
		
	return train_images, jnp.array(train_labels), test_images, jnp.array(test_labels)


