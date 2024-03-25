import tensorflow as tf
from tensorflow.experimental import numpy as tfnp


def take_along_axis(x, indices, axis=None):
    # TODO: move this workaround for dynamic shapes into keras-core.
    if axis < 0:
        axis = axis + indices.shape.rank
    # If all shapes after axis are 1, squeeze them off and use tf.gather.
    # tf.gather plays nicer with dynamic shapes in compiled functions.
    leftover_axes = list(range(axis + 1, indices.shape.rank))
    static_shape = indices.shape.as_list()
    squeezable = True
    for i in leftover_axes:
        if static_shape[i] != 1:
            squeezable = False
    if squeezable:
        if leftover_axes:
            indices = tf.squeeze(indices, leftover_axes)
        return tf.gather(x, indices, batch_dims=axis)
    # Otherwise, fall back to the tfnp call.
    return tfnp.take_along_axis(x, indices, axis=axis)
