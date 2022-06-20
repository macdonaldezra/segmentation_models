import typing as T
from ast import Mult
from typing import Tuple

from segmentation.models.components import double_conv
from tensorflow import Tensor
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Input,
    LayerNormalization,
    MultiHeadAttention,
    add,
)


def vision_mlp_block(x: Tensor, filter_nodes: T.List[int], activation: str = "gelu"):
    """
    Create a multi-layer perceptron block from the input tensor.

    Args:
        x Tensor: The input tensor.
        num_filters (int): A list that indicates the number of nodes in each multi-layer
        perceptron layer.The last layer must be such that its number of nodes is equal
        to the dimension of the key.
        activation (str, optional): A string corresponding to a valid Keras activation
        function. Defaults to "gelu".
    """
    for filter in filter_nodes:
        x = Dense(filter, activation=activation)(x)

    return x


def transformer_block(
    inputs: Tensor,
    attention_heads: int,
    key_dim: int,
    filter_nodes: T.List[int],
    activation: str = "gelu",
):
    """
    Create a Vision transform block of layers.

    Args:
        inputs (Tensor): The input tensor
        key_dim (int): _description_
    """
    x = inputs
    x = LayerNormalization()(x)
    x = MultiHeadAttention(num_heads=attention_heads, key_dim=key_dim)(x, x)

    x_add = add([x, inputs])

    # Multi-Layer Perceptron layer
    mlp = x_add
    mlp = LayerNormalization()(mlp)
    mlp = vision_mlp_block(mlp, filter_nodes, activation)

    return add([mlp, x_add])


def trans_unet(
    input_shape: Tuple[int, int, int],
    filters: int = 64,
    num_classes: int = 6,
    normalization_momentum: float = 0.15,
    dropout: float = 0.0,
    num_layers: int = 4,
    activation: str = "relu",
    kernel_initializer: str = "he_normal",
    transformer_activation: str = "gelu",
    name: str = "trans_unet",
) -> Model:
    inputs = Input(input_shape)
    x = inputs
    down_sampling_layers = []

    for _ in range(num_layers):
        x = double_conv(
            x,
            filters,
            activation=activation,
            momentum=normalization_momentum,
            dropout_rate=dropout,
            kernel_initializer=kernel_initializer,
        )
        down_sampling_layers.append(x)
        filters *= 2

    x = double_conv(
        x,
        filters,
        activation=activation,
        momentum=normalization_momentum,
        dropout_rate=dropout,
        kernel_initializer=kernel_initializer,
    )

    # Extract patches of downsampled images

    # Create vision transformer layers

    # Instantiate decoder layers

    # for _ in reversed(down_sampling_layers):
    #     filters //= 2
    #     x =

    outputs = Conv2D(num_classes, (1, 1), activation="relu")(x)

    return Model(inputs=[inputs], outputs=[outputs], name=name)
