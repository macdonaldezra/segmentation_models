import typing as T

from segmentation.models.components import double_conv
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Input,
    MaxPooling2D,
    concatenate,
)


def vanilla_unet(
    input_shape: T.Tuple[int, int, int],
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
    Returns a UNet model based on the UNet implementation provided here: https://arxiv.org/pdf/1505.04597.pdf

    Note that this model also includes dropout layers for both Conv2D layers, which differs from the implementation
    included in the provided paper. This implementation also does not crop images as
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
        x = concatenate([x, conv_layer])
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
