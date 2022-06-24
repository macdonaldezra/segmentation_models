import argparse
from pathlib import Path


def validate_filepath(path: Path) -> None:
    if not isinstance(path, Path):
        raise TypeError(f"{path} is not a valid filepath.")
    if not path.exists():
        raise FileNotFoundError(f"Path {path.as_posix()} does not exist.")


def parse_args() -> None:
    # Parse arguments for the following variables:
    # EPOCHS
    # Training file root directory
    # Output file root directory
    parser = argparse.ArgumentParser(
        description="Parse command line args for training a segmentation model."
    )
    parser.add_argument("-e", "--epochs", type=int, default=5)
    parser.add_argument("-d", "--data-directory", type=Path)
    parser.add_argument("-o", "--output-directory", type=Path)

    args = parser.parse_args()
    validate_filepath(args.data_directory)
    validate_filepath(args.output_directory)

    return args
