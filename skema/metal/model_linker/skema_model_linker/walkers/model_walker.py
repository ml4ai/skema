import abc
from abc import ABC


class ModelWalker(ABC):
    """ Defines the interface to walk through a __model to attach text extractions """

    def __iter__(self):
        return iter(self)

    @abc.abstractmethod
    def walk(self, callback=None, *args, **kwargs):
        pass
