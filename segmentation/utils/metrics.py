import tensorflow.keras.backend as K
from tensorflow import Tensor


def dice_coeffient(y_actual: Tensor, y_pred: Tensor, smooth: int = 1) -> float:
    """
    Also known as the F1 score, this metric is defined as two times
    the area of the intersection of y_actual and y_pred, divided by
    the sum of the areas of y_actual and y_pred.
    """
    y_actual_f = K.flatten(y_actual)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_actual_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        K.sum(y_actual_f) + K.sum(y_pred_f) + smooth
    )


def dice_coefficient_loss(y_actual: Tensor, y_pred: Tensor) -> float:
    """
    Compute the dice coefficient loss.
    """
    return 1 - dice_coeffient(y_actual, y_pred)


def jaccard_index(y_actual: Tensor, y_pred: Tensor, smooth: int = 1) -> float:
    """
    Compute intersection over union of two Tensors.
    """
    intersection = K.sum(y_actual * y_pred)
    union = K.sum(y_actual + y_pred)

    return (intersection + smooth) / (union - intersection + smooth)


def jaccard_distance(y_actual: Tensor, y_pred: Tensor, smooth: int = 1) -> float:
    """
    Compute the Jaccard Distance between two Tensors.
    """
    return (1 - jaccard_index(y_actual, y_pred, smooth)) * smooth
