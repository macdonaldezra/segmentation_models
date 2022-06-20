from typing import Tuple

from segmentation.models.components import concatenate_attention, double_conv
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D


def attention_unet(
    input_shape: Tuple[int, int, int],
    num_classes: int = 6,
    dropout: float = 0.0,
    normalization_momentum: float = 0.15,
    kernel_initializer: str = "he_normal",
    activation: str = "relu",
    name: str = "unet",
    filters: str = 64,
    num_layers: int = 4,
) -> Model:
    """
    UNet with single-headed attention layers.
    """

    inputs = Input(input_shape)
    x = inputs

    down_sampling_layers = []
    for _ in range(num_layers):
        x = double_conv(
            x,
            filters,
            momentum=normalization_momentum,
            dropout_rate=dropout,
            kernel_initializer=kernel_initializer,
            activation=activation,
        )
        down_sampling_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
        filters = filters * 2

    x = double_conv(
        x,
        filters,
        momentum=normalization_momentum,
        dropout_rate=dropout,
        kernel_initializer=kernel_initializer,
        activation=activation,
    )

    for conv_layer in reversed(down_sampling_layers):
        filters //= 2
        x = Conv2DTranspose(
            filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            kernel_initializer=kernel_initializer,
            padding="same",
        )(x)
        x = concatenate_attention(x, conv_layer)
        x = double_conv(
            x,
            filters,
            momentum=normalization_momentum,
            dropout_rate=dropout,
            kernel_initializer=kernel_initializer,
            activation=activation,
        )

    outputs = Conv2D(num_classes, (1, 1), activation="relu")(x)

    return Model(inputs=[inputs], outputs=[outputs], name=name)
