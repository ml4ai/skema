import sys
import platform
import random
import uuid

# -------------------------------------------
# Remove this block to generate different
# UUIDs everytime you run this code.
# This block should be right below the uuid
# import.
rd = random.Random()
uuid.uuid4 = lambda: uuid.UUID(int=rd.getrandbits(128))
# -------------------------------------------

def test_pygraphviz(error_message):
    """Tests whether the pygraphviz package is installed.
    If not, raises an exception"""

    if "pygraphviz" not in sys.modules:
        raise ModuleNotFoundError(
            "The pygraphviz package is not installed! "
            f"{error_message}"
        )


def choose_font():
    """Choose font for networkx graph labels, etc."""
    operating_system = platform.system()

    if operating_system == "Darwin":
        font = "Gill Sans"
    elif operating_system == "Windows":
        font = "Candara"
    else:
        font = "Ubuntu"

    return font
