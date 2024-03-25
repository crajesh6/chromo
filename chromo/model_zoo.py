import sys

import keras
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.util.tf_export import keras_export

from chromo import layers


@keras_export("keras.layers.Identity")
class Identity(Layer):
    """Identity layer.

    This layer should be used as a placeholder when no operation is to be
    performed. The layer is argument insensitive, and returns its `inputs`
    argument as output.

    Args:
        name: Optional name for the layer instance.
    """

    def call(self, inputs):
        return tf.nest.map_structure(tf.identity, inputs)


class AttentionPooling(keras.layers.Layer):
    def __init__(self, pool_size, *args, **kwargs):
        super(AttentionPooling, self).__init__(*args, **kwargs)
        self.pool_size = pool_size

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pool_size": self.pool_size,
            }
        )
        return config

    def build(self, input_shape):
        self.dense = keras.layers.Conv1D(
            filters=input_shape[-1], kernel_size=1, activation=None, use_bias=False
        )

    def call(self, inputs):
        N, L, F = inputs.shape
        inputs = tf.keras.layers.Cropping1D((0, L % self.pool_size))(inputs)
        inputs = tf.reshape(inputs, (-1, L // self.pool_size, self.pool_size, F))

        raw_weights = self.dense(inputs)
        att_weights = tf.nn.softmax(raw_weights, axis=-2)

        return tf.math.reduce_sum(inputs * att_weights, axis=-2)


class SELayer(tf.keras.layers.Layer):
    """
    Squeeze-and-Excitation (SE) Layer.

    Args:
        ratio (int): The ratio used to calculate the hidden units in the SE block.
    """

    def __init__(self, ratio, **kwargs):
        super(SELayer, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        _, _, c = input_shape
        self.pooling_layer = tf.keras.layers.GlobalAveragePooling1D()
        self.silu_layer = tf.keras.layers.Dense(
            c // self.ratio, activation="silu", use_bias=False
        )
        self.sigmoid_layer = tf.keras.layers.Dense(
            c, activation="sigmoid", use_bias=False
        )
        self.reshape_layer = tf.keras.layers.Reshape((1, c))

    def call(self, inputs):
        x = self.pooling_layer(inputs)
        x = self.silu_layer(x)
        x = self.sigmoid_layer(x)
        x = self.reshape_layer(x)
        return inputs * x


def create_motif_pooling_layer(motif_pooling_type, nn, filters, ratio=4):
    """
    Create a motif pooling layer based on the specified convolutional channel weight.

    Args:
        motif_pooling_type (str): The type of channel weighting.
        nn: The input tensor.
        filters (int): Number of convolutional filters.
        ratio (int): Ratio used in SE layer.

    Returns:
        Tensor: The output tensor after applying the specified motif pooling.
    """
    if motif_pooling_type == "se":
        nn = SELayer(ratio)(nn)
    elif motif_pooling_type == "softconv":
        nn = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
            use_bias=True,
            name="softconv_conv",
        )(nn)
        nn = keras.layers.Activation("relu", name="softconv_activation")(nn)
    return nn


def create_pooling_layer(pool_type, pool_size, name=None):
    """
    Create a pooling layer based on the specified pooling type.
    If the pool size is zero, this just returns the identity.

    Args:
        pool_type (str): The type of pooling ('max' or 'attention').
        pool_size (int): The size of the pooling window.
        name (str): The name of the layer.

    Returns:
        Keras Layer: The created pooling layer.
    """
    if pool_size == 0:
        return Identity(name=name)

    if pool_type == "attention_pool":
        return AttentionPooling(pool_size, name=name)
    elif pool_type == "max_pool":
        return tf.keras.layers.MaxPooling1D(pool_size, name=name)
    elif pool_type == "mean_pool":
        return tf.keras.layers.AveragePooling1D(pool_size, name=name)
    else:
        print("Unrecognized pooling layer. Abort")
        sys.exit(0)
    return


def create_conv_layer(conv_layer_type, filters, kernel_size, diag=None, offdiag=None):
    """
    Create a convolutional layer based on the specified convolution type.

    Args:
        conv_layer_type (str): The type of convolution ('standard' or 'pairwise').
        filters (int): Number of convolutional filters.
        kernel_size (int): Size of the convolutional kernel.
        diag: The diagonal regularization term for pairwise convolution.
        offdiag: The off-diagonal regularization term for pairwise convolution.

    Returns:
        Keras Layer: The created convolutional layer.
    """
    print(conv_layer_type, filters, kernel_size)
    if conv_layer_type == "standard":
        return keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            use_bias=True,
            name="conv1",
        )
    elif conv_layer_type == "pairwise":
        return layers.PairwiseConv1D(
            filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=layers.PairwiseReg(diag=diag, offdiag=offdiag),
            use_bias=True,
        )
    else:
        raise ValueError(f"Unsupported conv_layer_type: {conv_layer_type}")


def conv_layer(
    inputs,
    conv_layer_type=None,
    conv1_filters=None,
    conv1_kernel_size=None,
    conv1_activation=None,
    conv1_batchnorm=None,
    motif_pooling_type=None,
    spatial_pooling_type=None,
    conv1_pool_size=None,
    conv1_dropout=None,
):
    """
    Build a convolutional layer of the model.

    Args:
        input_shape: The shape of the input data.
        conv_layer_type (str): The type of convolutional layer.
        conv1_filters (int): Number of filters in the convolutional layer.
        conv1_kernel_size (int): Size of the convolutional kernel.
        conv1_activation (str): Activation function.
        conv1_batchnorm (bool): Whether to use batch normalization.
        motif_pooling_type (str): The type of channel weighting.
        spatial_pooling_type (str): The type of pooling.
        conv1_pool_size (int): The size of the pooling window.
        conv1_dropout (float): Dropout rate.

    Returns:
        Tensor: The output tensor of the convolutional layer.
    """
    diag = l2(1e-6)
    offdiag = l2(1e-3)

    # First conv. layer
    nn = create_conv_layer(
        conv_layer_type, conv1_filters, conv1_kernel_size, diag, offdiag
    )(inputs)
    if conv1_batchnorm:
        nn = tf.keras.layers.BatchNormalization(name="conv1_bn")(nn)
    nn = tf.keras.layers.Activation(conv1_activation, name="conv1_activation")(nn)
    nn = create_motif_pooling_layer(motif_pooling_type, nn, conv1_filters)
    nn = create_pooling_layer(spatial_pooling_type, conv1_pool_size, name="conv1_pool")(
        nn
    )
    nn = tf.keras.layers.Dropout(conv1_dropout, name="conv1_dropout")(nn)

    return nn


def create_base():
    """
    This function will modify the the base of the NN. That is, it will
    return either a NAM or a CAM version of the architecture.
    """
    pass


def create_output_layer(nn, output_shape, output_activation, branch_index=None):
    """
    Create the output layer of the model.

    Args:
        nn (tf.Tensor): Input tensor.
        output_shape (int): Number of output units.
        output_activation (str): Activation function for the output layer.

    Returns:
        tf.Tensor: Output tensor.
    """
    if output_activation == "linear":
        return keras.layers.Dense(
            output_shape,
            activation="linear",
            name=f"output_{branch_index}",
        )(nn)
    else:
        logits = keras.layers.Dense(
            output_shape,
            name=f"logits_{branch_index}",
        )(nn)
        return keras.layers.Activation(
            output_activation, name=f"output_{branch_index}"
        )(logits)


def create_dense_layers(
    nn, dense_units, dense_dropout, dense_activation, dense_batchnorm, branch_index=None
):
    """
    Create a stack of dense layers.

    Args:
        nn (tf.Tensor): Input tensor.
        dense_units (list): List of number of units in each dense layer.
        dense_dropout (list): List of dropout rates for each dense layer.
        dense_activation (str): Activation function for dense layers.
        dense_batchnorm (bool): Whether to use batch normalization.

    Returns:
        tf.Tensor: Output tensor.
    """
    dense_count = 0
    for units, dropout in zip(dense_units, dense_dropout):
        nn = keras.layers.Dense(units, name=f"dense_{dense_count}_{branch_index}")(nn)
        if dense_batchnorm:
            nn = keras.layers.BatchNormalization(
                name=f"bn_{dense_count}_{branch_index}"
            )(nn)
        nn = keras.layers.Activation(
            dense_activation, name=f"dense_activation_{dense_count}_{branch_index}"
        )(nn)
        nn = keras.layers.Dropout(
            dropout, name=f"dense_dropout_{dense_count}_{branch_index}"
        )(nn)
        dense_count += 1
    return nn


def create_mha_branch(
    nn,
    output_activation,
    output_shape,
    dense_units,
    dense_dropout,
    dense_activation,
    dense_batchnorm,
    mha_layernorm,
    mha_dropout,
    mha_heads,
    mha_d_model,
    branch_index=None,
):
    """
    Create a multi-head attention branch of the model.

    Args:
        nn (tf.Tensor): Input tensor.
        output_shape (int): Number of output units.
        dense_units (list): List of number of units in dense layers.
        dense_dropout (list): List of dropout rates for dense layers.
        dense_activation (str): Activation function for dense layers.
        dense_batchnorm (bool): Whether to use batch normalization.
        mha_layernorm (bool): Whether to use layer normalization for MHA.
        mha_dropout (float): Dropout rate for MHA.
        mha_heads (int): Number of heads in MHA.
        mha_d_model (int): Dimension of MHA.

    Returns:
        tf.Tensor: Output tensor.
    """
    if mha_layernorm:
        nn = keras.layers.LayerNormalization(name=f"mha_layernorm_{branch_index}")(nn)
    nn, att = layers.MultiHeadAttention(num_heads=mha_heads, d_model=mha_d_model)(
        nn, nn, nn
    )
    nn = keras.layers.Dropout(mha_dropout, name=f"mha_dropout_{branch_index}")(nn)

    nn = keras.layers.Flatten(name=f"flatten_{branch_index}")(nn)
    nn = create_dense_layers(
        nn,
        dense_units,
        dense_dropout,
        dense_activation,
        dense_batchnorm,
        branch_index=branch_index,
    )

    return create_output_layer(
        nn, output_shape, output_activation, branch_index=branch_index
    )


def create_nam_block(
    input_shape, inputs, filters, dense_units, activation, dropout_rate
):
    """
    Stacks two fully-connected layers with BatchNorm, activation, and Dropout.

    Args:
      inputs: Tensor representing the input to the stack.
      filters: Number of filters in the first fully-connected layer.
      dense_units: List of units for the two fully-connected layers.
      activation: Activation function to use.
      dropout_rate: Dropout rate to apply after each activation.

    Returns:
      Tensor representing the output of the layer stack.
    """
    N, L, A = inputs.shape

    nn = tf.keras.layers.Reshape((L, filters, 1))(inputs)  # N, L, F, 1

    nn = tf.keras.layers.EinsumDense(
        "abcd,de->abce", output_shape=[L, filters, dense_units[0]]
    )(nn)  # (N, L, F, D)
    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Activation(activation=activation)(nn)
    nn = tf.keras.layers.Dropout(rate=dropout_rate[0])(nn)
    print(f"Shape of nn: {nn.shape}")

    nn = tf.keras.layers.EinsumDense("abce,de->abcd", output_shape=[L, filters, 1])(
        nn
    )  # (N, L, F, 1)
    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Activation(activation=activation)(nn)
    nn = tf.keras.layers.Dropout(rate=dropout_rate[1])(nn)

    nn = tf.keras.layers.Reshape((L, filters))(nn)  # N, L, F
    nn = tf.keras.layers.GlobalAveragePooling1D()(nn)  # (N, F)

    return nn


def hominin_base_model(
    base_model,
    conv1_activation,
    conv1_batchnorm,
    motif_pooling_type,
    conv1_dropout,
    conv1_filters,
    conv1_kernel_size,
    spatial_pooling_type,
    conv1_pool_size,
    conv_layer_type,
    dense_activation,
    dense_batchnorm,
    dense_dropout,
    dense_units,
    input_shape,
    mha_d_model,
    mha_dropout,
    mha_head_type,
    mha_heads,
    mha_layernorm,
    output_activation,
    output_shape,
):
    """
    Build the entire model.

    Args:
        input_shape (tuple): Shape of the input tensor.
        output_shape (int): Number of output units.
        output_activation (str): Activation function for the output layer.
        mha_head_type (str): Type of multi-head attention head.

    Returns:
        tf.keras.Model: Built model.
    """

    keras.backend.clear_session()

    print(conv_layer_type, conv1_filters, conv1_kernel_size)
    # Create conv block
    inputs = keras.layers.Input(shape=input_shape, name="input")
    nn = conv_layer(
        inputs,
        conv_layer_type,
        conv1_filters,
        conv1_kernel_size,
        conv1_activation,
        conv1_batchnorm,
        motif_pooling_type,
        spatial_pooling_type,
        conv1_pool_size,
        conv1_dropout,
    )

    if base_model == "cam":
        # MHA and dense layers; create 1 branch per output head OR pooled
        if mha_head_type == "task_specific":
            outputs = [
                create_mha_branch(
                    nn,
                    output_activation,
                    1,
                    dense_units,
                    dense_dropout,
                    dense_activation,
                    dense_batchnorm,
                    mha_layernorm,
                    mha_dropout,
                    mha_heads,
                    mha_d_model,
                    branch_index=i,
                )
                for i in range(output_shape)
            ]

            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = create_mha_branch(
                nn,
                output_activation,
                output_shape,
                dense_units,
                dense_dropout,
                dense_activation,
                dense_batchnorm,
                mha_layernorm,
                mha_dropout,
                mha_heads,
                mha_d_model,
                branch_index=1,
            )

    elif base_model == "nam":
        nn = create_nam_block(
            input_shape=input_shape,
            inputs=nn,
            filters=conv1_filters,
            dense_units=dense_units,
            activation="relu",
            dropout_rate=dense_dropout,
        )

        outputs = create_output_layer(
            nn, output_shape, output_activation, branch_index=1
        )

    return tf.keras.Model(inputs=inputs, outputs=outputs)
