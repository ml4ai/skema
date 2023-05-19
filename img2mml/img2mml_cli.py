#!/usr/bin/env python

"""Command-line program to exercise the img2mml pipeline."""

import argparse
from img2mml.api import get_mathml_from_file


def get_mml(image_path) -> None:
    """
    Print rendered MML corresponding to a file
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        help="The path to the PNG file to process",
        default="tests/data/261.png",
    )

    args = parser.parse_args()
    mml = get_mathml_from_file(args.input)
    print(mml)
