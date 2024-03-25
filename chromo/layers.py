import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import tensor_shape
import tensorflow_probability as tfp

from chromo import ops


class PairwiseConv1D(keras.layers.Conv1D):
    """
    This class implement pairwise convolutions for 1D signals.  Standard
    convolutions implemented as `keras.layers.Conv1D` perform linear
    transformations of patches from the input signal. Pairwise convolutions
    perform linear transformations of all pairwise terms from entries in
    all patches of the input signal. The implementation is achieved by
    taking an outer product of each patch and performing a linear
    transformation of the pairwise (i.e. lower diagonal) terms. The rest of
    this docstring is copied from the keras.layers.Conv1D docstring.
    """

    __doc__ = __doc__ + super.__doc__
    padding_map_dict = {"same": "SAME", "valid": "VALID"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_patches(self, x):
        """
        x -> (N, L, A)
        This module is tested for correctness.
        """
        assert len(x.shape) == 3

        x = tf.expand_dims(x, axis=2)  ## (N, L, 1, A)
        sizes = [1, self.kernel_size[0], 1, 1]
        strides = [1, self.strides[0], 1, 1]
        rates = [1, self.dilation_rate[0], 1, 1]
        padding = self.padding_map_dict[self.padding]
        xp = tf.image.extract_patches(
            x, sizes=sizes, strides=strides, rates=rates, padding=padding
        )
        xp = tf.squeeze(xp, axis=2)  ## (N, num patches, flattened patch size)
        return xp

    def _outer_product(self, xpatches):
        """
        xpatches -> (N, numpatches, patch size*A)
        RETURNS:
        xout -> (N, num patches, patch size*A, patch_size*A)
        """
        res = tf.einsum(
            "ijk, ijl -> ijkl", xpatches, xpatches
        )  ## (N, numpatches, P*A, P*A)
        res = tf.linalg.set_diag(
            res, diagonal=xpatches
        )  ## replace the sq. term with unit power in the diag.
        return res

    @property
    def full_kernel(self):
        k = tf.transpose(self.kernel, [1, 0])  ## (C, numweights,)
        k = tfp.math.fill_triangular(k)  ## (C, P*A, P*A)
        k = tf.transpose(k, [1, 2, 0])  ## (P*A, P*A, C)
        return k

    @property
    def diag_kernel(self):
        """
        Returns the diagonal of the kernel.
        """
        k = self.full_kernel  ## (P*A, P*A, C)
        k = tf.transpose(k, [2, 0, 1])  ## (C, P*A, P*A)
        k = tf.linalg.diag_part(k)  ## (C, P*A)
        k = tf.transpose(k, [1, 0])  ## (P*A, C)
        return k

    def build(self, input_shape):
        """
        Expected input_shape is (N, L, A)
        """
        input_shape = tensor_shape.TensorShape(input_shape)  # (L, A)
        A = input_shape[-1]  # A
        P = self.kernel_size[0]  # P
        flat_patch_size = P * A
        kernel_shape = [
            int(flat_patch_size * (flat_patch_size + 1) * 0.5),
            self.filters,
        ]  ## (numweights, C)

        # add the kernel
        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype,
        )

        # add the bias
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        """
        inputs -> (N, L, in_channels)
        RETURNS:
        outputs -> (N, L, out_channels)
        """

        xp = self._get_patches(inputs)  ## (N, numpatches, P*A)

        # take the outer product
        xout = self._outer_product(xp)  ## (N, numpatches, P*A, P*A)

        # compute the output
        kern = self.full_kernel  ## (P*A, P*A, C)
        outputs = tf.einsum("ijkl, klm -> ijm", xout, kern)

        # add the bias
        if self.use_bias:
            outputs = outputs + self.bias

        # apply activation function
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


class PairwiseReg(keras.regularizers.Regularizer):
    """

    A regularizer than applies separate regularization functions on
    the diagonal and off-diagonal terms in the pairwise kernel.
    """

    def __init__(self, diag, offdiag, *args, **kwargs):
        """
        diag_regularizer <keras.regularizer.Regularizer> - The
        offdiag_regularizer <keras.regularizer.Regularizer>
        """
        super().__init__(*args, **kwargs)
        self.diag = diag
        self.offdiag = offdiag

    def __call__(self, x):
        """
        x -> Pairwise kernel (expected shape = (numterms, A, C))
        """
        ndims = len(x.shape)
        perm = list(np.arange(1, ndims)) + [0]
        x = tf.transpose(x, perm)  ## move 1st dimension to the end
        x = tfp.math.fill_triangular(x)

        diag_part = tf.linalg.diag_part(x)  ##(A, C, P)
        offdiag_part = x - tf.linalg.diag(diag_part)  ## (A, C, P, P)

        res1 = self.diag(diag_part)
        res2 = self.offdiag(offdiag_part)
        return res1 + res2

    def get_config(self):
        config = {}
        config["diag"] = self.diag
        config["offdiag"] = self.offdiag
        return config


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads, embedding_size=None):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.embedding_size = d_model if embedding_size is None else embedding_size

        assert d_model % self.num_heads == 0 and d_model % 6 == 0

        self.depth = d_model // self.num_heads

        self.wq = keras.layers.Dense(d_model, use_bias=False)
        self.wk = keras.layers.Dense(d_model, use_bias=False)
        self.wv = keras.layers.Dense(d_model, use_bias=False)

        self.r_k_layer = keras.layers.Dense(d_model, use_bias=False)
        self.r_w = tf.Variable(
            tf.random_normal_initializer(0, 0.5)(
                shape=[1, self.num_heads, 1, self.depth]
            ),
            trainable=True,
        )
        self.r_r = tf.Variable(
            tf.random_normal_initializer(0, 0.5)(
                shape=[1, self.num_heads, 1, self.depth]
            ),
            trainable=True,
        )

        self.dense = keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size, seq_len):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]
        seq_len = tf.constant(q.shape[1])

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(
            q, batch_size, seq_len
        )  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(
            k, batch_size, seq_len
        )  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(
            v, batch_size, seq_len
        )  # (batch_size, num_heads, seq_len_v, depth)
        q = q / tf.math.sqrt(tf.cast(self.depth, dtype=tf.float32))

        pos = tf.range(-seq_len + 1, seq_len, dtype=tf.float32)[tf.newaxis]
        feature_size = self.embedding_size // 6

        seq_length = tf.cast(seq_len, dtype=tf.float32)
        exp1 = f_exponential(tf.abs(pos), feature_size, seq_length=seq_length)
        exp2 = tf.multiply(exp1, tf.sign(pos)[..., tf.newaxis])
        cm1 = f_central_mask(tf.abs(pos), feature_size, seq_length=seq_length)
        cm2 = tf.multiply(cm1, tf.sign(pos)[..., tf.newaxis])
        gam1 = f_gamma(tf.abs(pos), feature_size, seq_length=seq_length)
        gam2 = tf.multiply(gam1, tf.sign(pos)[..., tf.newaxis])

        # [1, 2seq_len - 1, embedding_size]
        positional_encodings = tf.concat([exp1, exp2, cm1, cm2, gam1, gam2], axis=-1)
        positional_encodings = keras.layers.Dropout(0.1)(positional_encodings)

        # [1, 2seq_len - 1, d_model]
        r_k = self.r_k_layer(positional_encodings)

        # [1, 2seq_len - 1, num_heads, depth]
        r_k = tf.reshape(r_k, [r_k.shape[0], r_k.shape[1], self.num_heads, self.depth])
        r_k = tf.transpose(r_k, perm=[0, 2, 1, 3])
        # [1, num_heads, 2seq_len - 1, depth]

        # [batch_size, num_heads, seq_len, seq_len]
        content_logits = tf.matmul(q + self.r_w, k, transpose_b=True)

        # [batch_size, num_heads, seq_len, 2seq_len - 1]
        relative_logits = tf.matmul(q + self.r_r, r_k, transpose_b=True)
        # [batch_size, num_heads, seq_len, seq_len]
        relative_logits = relative_shift(relative_logits)

        # [batch_size, num_heads, seq_len, seq_len]
        logits = content_logits + relative_logits
        attention_map = tf.nn.softmax(logits)

        # [batch_size, num_heads, seq_len, depth]
        attended_values = tf.matmul(attention_map, v)
        # [batch_size, seq_len, num_heads, depth]
        attended_values = tf.transpose(attended_values, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(
            attended_values, [batch_size, seq_len, self.d_model]
        )

        output = self.dense(concat_attention)

        return output, attention_map


# ------------------------------------------------------------------------------------------
# Positional encoding functions for Multi-Head Attention
# ------------------------------------------------------------------------------------------


def f_exponential(positions, feature_size, seq_length=None, min_half_life=3.0):
    if seq_length is None:
        seq_length = tf.cast(tf.reduce_max(tf.abs(positions)) + 1, dtype=tf.float32)
    max_range = tf.math.log(seq_length) / tf.math.log(2.0)
    half_life = tf.pow(2.0, tf.linspace(min_half_life, max_range, feature_size))
    half_life = tf.reshape(
        half_life, shape=[1] * positions.shape.rank + half_life.shape
    )
    positions = tf.abs(positions)
    outputs = tf.exp(-tf.math.log(2.0) / half_life * positions[..., tf.newaxis])
    return outputs


def f_central_mask(positions, feature_size, seq_length=None):
    center_widths = tf.pow(2.0, tf.range(1, feature_size + 1, dtype=tf.float32)) - 1
    center_widths = tf.reshape(
        center_widths, shape=[1] * positions.shape.rank + center_widths.shape
    )
    outputs = tf.cast(center_widths > tf.abs(positions)[..., tf.newaxis], tf.float32)
    return outputs


def f_gamma(positions, feature_size, seq_length=None):
    if seq_length is None:
        seq_length = tf.reduce_max(tf.abs(positions)) + 1
    stdv = seq_length / (2 * feature_size)
    start_mean = seq_length / feature_size
    mean = tf.linspace(start_mean, seq_length, num=feature_size)
    mean = tf.reshape(mean, shape=[1] * positions.shape.rank + mean.shape)
    concentration = (mean / stdv) ** 2
    rate = mean / stdv**2

    def gamma_pdf(x, conc, rt):
        log_unnormalized_prob = tf.math.xlogy(concentration - 1.0, x) - rate * x
        log_normalization = tf.math.lgamma(concentration) - concentration * tf.math.log(
            rate
        )
        return tf.exp(log_unnormalized_prob - log_normalization)

    probabilities = gamma_pdf(
        tf.abs(tf.cast(positions, dtype=tf.float32))[..., tf.newaxis],
        concentration,
        rate,
    )
    outputs = probabilities / tf.reduce_max(probabilities)
    return outputs


def relative_shift(x):
    to_pad = tf.zeros_like(x[..., :1])
    x = tf.concat([to_pad, x], -1)
    _, num_heads, t1, t2 = x.shape
    x = tf.reshape(x, [-1, num_heads, t2, t1])
    x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, [-1, num_heads, t1, t2 - 1])
    x = tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, (t2 + 1) // 2])
    return x


# ------------------------------------------------------------------------------------------
# Rotary Positional Encoding
# ------------------------------------------------------------------------------------------
# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from keras_nlp.api_export import keras_nlp_export
# import keras


# @keras_nlp_export("keras_nlp.layers.RotaryEmbedding")
class RotaryEmbedding(keras.layers.Layer):
    """Rotary positional encoding layer.

    This layer encodes absolute positional information with a rotation
    matrix. It calculates the rotary encoding with a mix of sine and
    cosine functions with geometrically increasing wavelengths.
    Defined and formulated in [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v4).
    The input must be a tensor with shape a sequence dimension and a feature
    dimension. Typically, this will either an input with shape
    `(batch_size, sequence_length, feature_length)` or
    `(batch_size, sequence_length, num_heads, feature_length)`.
    This layer will return a new tensor with the rotary embedding applied to
    the input tensor.

    Args:
        max_wavelength: int. The maximum angular wavelength of the sine/cosine
            curves.
        scaling_factor: float. The scaling factor used to scale frequency range.
        sequence_axis: int. Sequence axis in the input tensor.
        feature_axis: int. Feature axis in the input tensor.

    Call arguments:
        inputs: The tensor inputs to apply the embedding to. This can have
            any shape, but must contain both a sequence and feature axis. The
            rotary embedding will be applied to `inputs` and returned.
        start_index: An integer or integer tensor. The starting position to
            compute the rotary embedding from. This is useful during cached
            decoding, where each position is predicted separately in a loop.

    Examples:

    ```python
    batch_size = 16
    feature_length = 18
    sequence_length = 256
    num_heads = 8

    # No multi-head dimension.
    tensor = np.ones((batch_size, sequence_length, feature_length))
    rot_emb_layer = RotaryEmbedding()
    tensor_rot = rot_emb_layer(tensor)

    # With multi-head dimension.
    tensor = np.ones((batch_size, sequence_length, num_heads, feature_length))
    tensor_rot = rot_emb_layer(tensor)
    ```

    References:
     - [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v4)
    """

    def __init__(
        self,
        max_wavelength=10000,
        scaling_factor=1.0,
        sequence_axis=1,
        feature_axis=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_wavelength = max_wavelength
        self.sequence_axis = sequence_axis
        self.feature_axis = feature_axis
        self.scaling_factor = scaling_factor
        self.built = True

    def call(self, inputs, start_index=0):
        cos_emb, sin_emb = self._compute_cos_sin_embedding(inputs, start_index)
        return self._apply_rotary_pos_emb(inputs, cos_emb, sin_emb)

    def _apply_rotary_pos_emb(self, tensor, cos_emb, sin_emb):
        x1, x2 = ops.split(tensor, 2, axis=self.feature_axis)
        half_rot_tensor = ops.concatenate((-x2, x1), axis=self.feature_axis)
        return (tensor * cos_emb) + (half_rot_tensor * sin_emb)

    def _compute_cos_sin_embedding(self, inputs, start_index=0):
        def get_axis(axis):
            return axis if axis > 0 else len(inputs.shape) + axis

        feature_axis = get_axis(self.feature_axis)
        sequence_axis = get_axis(self.sequence_axis)

        rotary_dim = ops.shape(inputs)[feature_axis]
        inverse_freq = self._get_inverse_freq(rotary_dim)

        seq_len = ops.shape(inputs)[self.sequence_axis]
        tensor = ops.cast(ops.arange(seq_len), self.compute_dtype) + start_index

        tensor = ops.cast(tensor, dtype=inverse_freq.dtype)
        freq = ops.einsum("i,j->ij", tensor, inverse_freq)
        embedding = ops.concatenate((freq, freq), axis=-1)

        # Reshape the embedding to be broadcastable with input shape.
        if feature_axis < sequence_axis:
            embedding = ops.transpose(embedding)
        for axis in range(len(inputs.shape)):
            if axis != sequence_axis and axis != feature_axis:
                embedding = ops.expand_dims(embedding, axis)

        return ops.cos(embedding), ops.sin(embedding)

    def _get_inverse_freq(self, rotary_dim):
        freq_range = ops.arange(0, rotary_dim, 2)
        freq_range = ops.cast(freq_range, self.compute_dtype)
        freq_range = freq_range / ops.cast(self.scaling_factor, self.compute_dtype)
        inverse_freq = 1.0 / (
            self.max_wavelength
            ** (freq_range / ops.cast(rotary_dim, self.compute_dtype))
        )
        return inverse_freq

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_wavelength": self.max_wavelength,
                "scaling_factor": self.scaling_factor,
                "sequence_axis": self.sequence_axis,
                "feature_axis": self.feature_axis,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
