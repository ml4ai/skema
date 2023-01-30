#!/usr/bin/env python

"""Example Python client program to communicate with the skema-py service."""

import os
import json
import requests
import argparse


def system_to_json(
    root_path: str, system_filepaths: str, system_name: str
) -> str:
    files = []
    blobs = []

    with open(system_filepaths, "r") as f:
        files = f.readlines()
    files = [file.strip() for file in files]

    for file_path in files:
        full_path = os.path.join(root_path, file_path)
        with open(full_path, "r") as f:
            blobs.append(f.read())

    root_name = os.path.basename(os.path.normpath(root_path))

    return json.dumps(
        {
            "files": files,
            "blobs": blobs,
            "system_name": system_name,
            "root_name": root_name,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://localhost:8000/fn-given-filepaths",
        help="Host machine where the Code2FN service is running",
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "If this flag is provided, the program writes the response "
            "to a file. Otherwise it prints the response to standard output."
        ),
    )

    parser.add_argument("root_path", type=str)
    parser.add_argument("system_filepaths", type=str)
    parser.add_argument("system_name", type=str)

    args = parser.parse_args()

    data = system_to_json(
        args.root_path, args.system_filepaths, args.system_name
    )
    response = requests.post(args.url, data=data)

    if args.write:
        with open(f"{args.system_name}--Gromet-FN-auto.json", "w") as f:
            f.write(response.json())
    else:
        print(response.json())
