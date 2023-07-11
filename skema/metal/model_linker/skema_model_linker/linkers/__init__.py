heuristics = {
    "s": ["susceptible"],
    "d": ["diagnosed", "deceased"],
    "a": ["aligned"],
    "r": ["recovered"],
    "i": ["infected"],
    "h": ["healed"],
    "e": ['exposed', 'extinct']
}

from .amr_linker import AMRLinker
from .petrinet import PetriNetLinker
from .regnet import RegNetLinker
from .gromet_linker import GrometLinker



__all__ = [
    "AMRLinker",
    "PetriNetLinker",
    "RegNetLinker",
    "GrometLinker",
]