#!/usr/bin/env python

"""Example Python client program to work with the img2mml web service."""

import argparse
import requests


def get_mml(image_path: str, url: str) -> str:
    """
    It sends the http requests to put in an image to translate it into MathML.
    """
    with open(image_path, "rb") as f:
        r = requests.put(url, files={"file": f})
    return r.text


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

    parser.add_argument(
        "--url",
        help="The URL to the img2mml service endpoint.",
        default="http://localhost:8000/get-mml",
    )

    args = parser.parse_args()
    mml = get_mml(args.input, args.url)
    print(mml)
