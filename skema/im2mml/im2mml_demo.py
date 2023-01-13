"""Example Python client program to work with the im2mml web service."""

import argparse
import requests


def get_mml(image_path: str) -> str:
    """
    It sends the http requests to put in an image to translate it into MathML.
    """
    with open(image_path, "rb") as f:
        r = requests.put("http://127.0.0.1:8000/get-mml", files={"file": f})
    return r.text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input",
        help="The path to the PNG file to process",
        default="tests/data/261.png"
    )

    mml = get_mml("tests/data/261.png")
    print(mml)
