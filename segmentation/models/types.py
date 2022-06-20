import typing as T

from pydantic import BaseModel


class Conv2DParams(BaseModel):
    name: str
    dropout: float
    momentum: float
    kernel_initializer: str
    activation: str
    filters: int
    num_layers: int
    bias: bool = False

    def layer_params(self) -> T.Dict[str, T.Union[str, float, int]]:
        """
        Return parameters that map to keras Conv2D layers.
        """
        return {
            "dropout": self.dropout,
            "momentum": self.momentum,
            "kernel_initializer": self.kernel_initializer,
            "activation": self.activation,
            "bias": self.bias,
        }
