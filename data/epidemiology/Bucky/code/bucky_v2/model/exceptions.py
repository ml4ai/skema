"""Some custom bucky related exceptions."""
from ..exceptions import BuckyException


class SimulationException(BuckyException):
    """A generic exception to throw when there's an error related to the simulation."""

    pass  # pylint: disable=unnecessary-pass


class StateValidationException(SimulationException):
    """Thrown when the state vector is in an invalid state."""

    pass  # pylint: disable=unnecessary-pass
