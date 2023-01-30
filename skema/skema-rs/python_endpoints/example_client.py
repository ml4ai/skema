#!/usr/bin/env python

"""Example Python client program to communicate with the Code2FN service."""

import os
import json
import requests
import argparse


if __name__ == "__main__":
    with open('../../../data/demo/CHIME_SVIIvR_core--Gromet-FN-auto.json') as f:
        r = requests.post(f"http://localhost:8080/models", json=json.load(f))
        MODEL_ID = r.json()


    # Get opis and opos
    request_url = f"http://localhost:8080/models/{MODEL_ID}/named_ports"
    response = requests.post(request_url)
    print(response.json())

    # Get other thing
    obj = response.json()
    request_url = f"http://localhost:8000/get-pyacset"
    response = requests.post(request_url, json=obj)
    print(response.json())

