"""Generic utility functions/classes used in the model."""

# TODO break into files now that we have the utils submodule (also update __init__)

###CTM import copy
import datetime
###CTM import logging
from functools import lru_cache

###CTM import numpy as np
###CTM import pandas as pd
###CTM import tqdm

###CTM from .. import __version__
###CTM_START
__version__ = "1.0.0.alpha0.post2"
###CTM_END


@lru_cache(maxsize=None)
def generate_runid():
    """Gets a UUID based of the current datatime and caches it."""
    dt_now = datetime.datetime.now()
    return str(dt_now).replace(" ", "__").replace(":", "_").split(".", maxsplit=1)[0]


def _banner(msg=None):
    """A banner for the CLI."""
    print(r" ____             _          ")  # noqa: T201
    print(r"| __ ) _   _  ___| | ___   _ ")  # noqa: T201
    print(r"|  _ \| | | |/ __| |/ / | | |", end="")  # noqa: T201
    print(f"   v{__version__}")  # noqa: T201
    print(r"| |_) | |_| | (__|   <| |_| |")  # noqa: T201
    print(r"|____/ \__,_|\___|_|\_\\__, |", end="")  # noqa: T201
    print(f"   {msg}" if msg is not None else "")  # noqa: T201
    print(r"                       |___/ ")  # noqa: T201
    print(r"                             ")  # noqa: T201
