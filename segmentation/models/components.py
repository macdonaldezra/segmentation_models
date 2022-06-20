from tensorflow import Tensor
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dropout,
    add,
    concatenate,
    multiply,
)


def double_conv(
    inputs: Tensor,
    channels: int,
    momentum: float = 0.15,
    dropout_rate: float = 0.0,
    kernel_initializer: str = "he_normal",
    activation: str = "relu",
):
    """
    Create a tensor with two Conv2D layers with dropout and batch normalization.
    """

    x = Conv2D(
        channels,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        activation=activation,
    )(inputs)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(
        channels,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        activation=activation,
    )(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization(momentum=momentum)(x)

    return x


def attention_layer(up_layer: Tensor, down_layer: Tensor, filters: int):
    """
    Create an attention layer

    Args:
        up_layer (Tensor): The Up-Sampling layer
        down_layer (Tensor): The Down-Sampling layer
        filters (int): The number of filters to be used for each layers Conv2D layer.

    Returns:
        A Keras tensor.
    """

    up_block = Conv2D(
        filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(up_layer)
    down_block = Conv2D(
        filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(down_layer)

    f = Activation("relu")(add([up_block, down_block]))
    g = Conv2D(
        filters=1,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(f)
    h = Activation("sigmoid")(g)

    return multiply([down_layer, h])


def concatenate_attention(up_layer: Tensor, down_layer: Tensor):
    """
    Creates an attention layer concatenated with the upsample layer and concat
    """
    filters = up_layer.get_shape().as_list()[-1]
    attention = attention_layer(up_layer, down_layer, filters)

    return concatenate([down_layer, attention])
