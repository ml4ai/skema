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
    request_url = f"http://localhost:8000/endpoint1"
    response = requests.post(request_url, json=MODEL_ID)
    print(response.json())

    # Get other thing
    obj = response.json()
    data = {"opis": obj["opis"], "opos": obj["opos"]}
    request_url = f"http://localhost:8000/endpoint2"
    response = requests.post(request_url, json=data)
    print(response.json())
    