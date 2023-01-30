#!/usr/bin/env python

"""Example Python client program to communicate with the skema-py service."""

import os
import json
from requests import get, post, delete
import argparse


if __name__ == "__main__":
    with open("11c--Gromet-FN-manual-dynamics.json") as f:
        r = post(f"http://localhost:8080/models", json=json.load(f))
        MODEL_ID = r.text

    # Get opis and opos
    request_url = f"http://localhost:8080/models/{MODEL_ID}/named_ports"
    named_ports = get(request_url).json()
    print(named_ports)

    # Get other thing
    request_url = f"http://localhost:8000/get-pyacset"
    r = post(request_url, json=named_ports)
    print(r.json())

    # Delete the model so we don't waste space
    delete(f"http://localhost:8080/models/{MODEL_ID}")
