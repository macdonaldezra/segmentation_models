import random
import typing as T
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from segmentation.models.attention_unet import attention_unet
from segmentation.train.utils import parse_args
from segmentation.utils.metrics import (dice_coefficient_loss, dice_coeffient,
                                        jaccard_index)
from sklearn.model_selection import train_test_split


def save_history(filepath: Path, history: T.Any, file_format: str = "csv") -> None:
    """
    Save training history as a JSON file.
    """
    date_obj = datetime.now()
    date_str = date_obj.strftime("%d-%b-%Y-%H:%M")
    history_dir = filepath / "history"
    history_dir.mkdir(exist_ok=True, parents=True)

    hist_df = pd.DataFrame(history.history)

    if file_format == "csv":
        with open(
            history_dir / ("train_history_" + date_str + ".csv"), "w"
        ) as out_file:
            hist_df.to_csv(out_file)
    else:
        with open(
            history_dir / ("train_history_" + date_str + ".json"), "w"
        ) as out_file:
            hist_df.to_json(out_file)


def set_seed(seed: int = random.randint(0, 1000000)):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def scale_image_and_binary_mask(
    image: np.ndarray, mask: np.ndarray
) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Scale an image and mask between 0 and 1. Binary 1 or 0 is applied to mask image.
    """
    image = image / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask < 0.5] = 0

    return (image, mask)


def training_generator(
    df: pd.DataFrame,
    generator_params: T.Dict[str, T.Any],
    batch_size: int = 64,
    image_color_mode="rgb",
    mask_color_mode="grayscale",
    image_save_prefix="image",
    mask_save_prefix="mask",
    save_to_dir=None,
    target_size=(256, 256),
    seed=1,
) -> T.Generator:
    """
    Generator function that yields images and masks that have been modified based on the
    provided keras Image Generator's parameters.
    """
    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        **generator_params
    )
    mask_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        **generator_params
    )

    image_generator = image_data_generator.flow_from_dataframe(
        df,
        x_col="filename",
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
    )
    mask_generator = mask_data_generator.flow_from_dataframe(
        df,
        x_col="mask",
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
    )

    train_gen = zip(image_generator, mask_generator)
    for (image, mask) in train_gen:
        image, mask = scale_image_and_binary_mask(image, mask)
        yield (image, mask)


def train_attention_unet(
    epochs: int, data_directory: Path, output_directory: Path
) -> T.Dict[str, T.Any]:
    """
    Load training and test set and train the attention model on segmenting Brain MRI data.
    """
    set_seed()

    # declare filepaths for mask and training files
    mri_directory = data_directory / "brain_mri"
    mask_files = list(mri_directory.glob("lgg-mri-segmentation/kaggle_3m/*/*_mask*"))
    mask_files = list(map(str, mask_files))
    training_files = [path.replace("_mask", "") for path in mask_files]

    df = pd.DataFrame(data={"filename": training_files, "mask": mask_files})
    df_train, df_test = train_test_split(df, test_size=0.1)
    df_train, df_val = train_test_split(df_train, test_size=0.2)

    BATCH_SIZE = 32
    IMAGE_WIDTH = 256
    IMAGE_HEIGHT = 256

    IMAGE_GENERATOR_PARAMS = {
        "rotation_range": 0.2,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "shear_range": 0.05,
        "zoom_range": 0.05,
        "horizontal_flip": True,
        "fill_mode": "nearest",
    }

    train_generator = training_generator(
        df_train,
        IMAGE_GENERATOR_PARAMS,
        BATCH_SIZE,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    )
    test_generator = training_generator(
        df_test, {}, BATCH_SIZE, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
    )

    optimizer = tf.keras.optimizers.Adam(
        beta_1=0.9,
        beta_2=0.999,
        epsilon=None,
        amsgrad=False,
    )

    training_path = output_directory / "training"
    training_path.mkdir(parents=True, exist_ok=True)

    callbacks = [
        # Save training weights only for model that has highest accuracy on validation dataset
        tf.keras.callbacks.ModelCheckpoint(
            (training_path / "unet_brain_mri_segmentation.hdf5").as_posix(),
            monitor="val_binary_accuracy",
            verbose=1,
            save_best_only=True,
            mode="max",
        ),
        # Stop training model after 5 epochs if the model accuracy doesn't improve.
        tf.keras.callbacks.EarlyStopping(monitor="val_binary_accuracy", patience=5),
        # Reduce learning rate if accuracy is not improving on training dataset.
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="binary_accuracy", factor=0.1, patience=3
        ),
        # Stop training process if we are returning NaN value in loss
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    model = attention_unet(
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), num_classes=1, num_layers=1
    )
    model.compile(
        optimizer=optimizer,
        loss=dice_coefficient_loss,
        metrics=[
            "binary_accuracy",
            jaccard_index,
            dice_coeffient,
        ],
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=len(df_train) / BATCH_SIZE,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=test_generator,
        validation_steps=len(df_val) / BATCH_SIZE,
    )

    save_history(training_path, history)


if __name__ == "__main__":
    if len(tf.config.list_physical_devices("GPU")) < 1:
        print("No GPU available, not running job.")
        exit(1)

    args = parse_args()

    # train_attention_unet(args.epochs, args.data_directory, args.output_directory)
