# -*- coding: utf-8 -*-

from typing import List
from fastapi import FastAPI, File
from skema.img2mml.api import get_mathml_from_bytes


# Create a web app using FastAPI

app = FastAPI()


@app.get("/ping", summary="Ping endpoint to test health of service")
def ping():
    return "The img2mml service is running."

@app.put("/get-mml", summary="Get MathML representation of an equation image")
async def get_mathml(file: bytes = File()):
    """
    Endpoint for generating MathML from an input image.
    """
    # convert png image to tensor
    return get_mathml_from_bytes(file)
