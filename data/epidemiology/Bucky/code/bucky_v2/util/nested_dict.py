"""Provide a nested version of dict() along with convenient API additions (apply, update, etc)."""
from collections import OrderedDict
from collections.abc import Collection, Mapping, MutableMapping  # pylint: disable=no-name-in-module
from copy import deepcopy
from pprint import pformat

from ruamel.yaml import YAML
from ruamel.yaml.representer import RepresenterError

yaml = YAML()


def _is_dict_type(x):
    """Check is a variable has a dict-like interface."""
    return isinstance(x, MutableMapping)


def _is_list_type(x):
    """Check is a variable has a list-like interface."""
    return isinstance(x, Collection) and not isinstance(x, str)


class NestedDict(MutableMapping):
    """A nested version of dict."""

    def __init__(self, dict_of_dicts=None, seperator=".", ordered=True):
        """Init an empty dict or convert a dict of dicts into a NestedDict."""
        # TODO detect flattened dicts too?
        self.seperator = seperator  # TODO double check this is being enforced EVERYWHERE...
        self.ordered = ordered
        if dict_of_dicts is not None:
            if not _is_dict_type(dict_of_dicts):
                raise TypeError
            self._data = self.from_dict(dict_of_dicts)
        else:
            self._data = OrderedDict() if self.ordered else {}

    def __setitem__(self, key, value):
        """Setitem, doing it recusively if a flattened key is used."""
        if isinstance(key, int):
            key = str(key)
        elif not isinstance(key, str):
            raise NotImplementedError(f"{self.__class__} only supports str-typed keys; for now")

        if self.seperator in key:
            keys = key.split(self.seperator)
            last_key = keys.pop()
            try:
                tmp = self._data
                for k in keys:
                    if k in tmp:
                        tmp = tmp[k]
                    else:
                        tmp[k] = self.__class__(seperator=self.seperator, ordered=self.ordered)
                        tmp = tmp[k]
                tmp[last_key] = value
            except KeyError as err:
                raise KeyError(key) from err
        else:
            self._data[key] = value

    def __getitem__(self, key):
        """Getitem, supports flattened keys."""
        if isinstance(key, int):
            key = str(key)
        elif not isinstance(key, str):
            raise NotImplementedError(f"{self.__class__} only supports str-typed keys; for now")

        if self.seperator in key:
            keys = key.split(self.seperator)
            try:
                tmp = self._data
                for k in keys:
                    if k in tmp:
                        tmp = tmp[k]
            except KeyError as err:
                raise KeyError(key) from err
            return tmp
        else:
            return self._data[key]

    def __delitem__(self, key):
        """WIP."""
        # TODO this doesnt work for flattened keys
        del self._data[key]

    def __iter__(self):
        """WIP."""
        # provide a flatiter too?
        return iter(self._data)

    def __len__(self):
        """WIP."""
        # TODO
        return len(self._data)

    def __repr__(self):
        """REPL string representation for NestedDict, basically just yaml-ize it."""
        # Just lean on yaml for now but it makes arrays very ugly
        try:
            return self.to_yaml()
        except RepresenterError:
            # Fallback to printing a dict if something prevents yaml
            return pformat(self.to_dict())

    # def __str__(self):

    def flatten(self, parent=""):
        """Flatten to a normal dict where the hierarchy exists in the key names."""
        ret = OrderedDict() if self.ordered else {}

        def _recursive_flatten(v, parent_key=""):
            """Recursively flatten, handling both list and dict types."""
            if _is_list_type(v):
                for i, v2 in enumerate(v):
                    key = parent_key + self.seperator + str(i) if parent_key else str(i)
                    _recursive_flatten(v2, key)
            elif _is_dict_type(v):
                for k, v2 in v.items():
                    key = parent_key + self.seperator + k if parent_key else k
                    _recursive_flatten(v2, key)
            else:
                ret[parent_key] = v

        _recursive_flatten(self)

        return ret

    def from_flat_dict(self, flat_dict):
        """Create a NestedDict from a flattened dict."""
        ret = self.__class__()
        for k, v in flat_dict.items():
            ret[k] = v
        return ret

    def from_dict(self, dict_of_dicts):
        """Create a NestedDict from a dict of dicts."""
        ret = self.__class__(seperator=self.seperator, ordered=self.ordered)
        for k, v in dict_of_dicts.items():
            if _is_dict_type(v):
                ret[k] = self.from_dict(v)
            else:
                ret[k] = v
        return ret

    # def from_yaml(f):

    def to_dict(self):
        """Return self but as a proper dict of dicts."""
        ret = {}
        for k, v in self.items():
            if _is_dict_type(v):  # isinstance(v, type(self)):
                ret[k] = v.to_dict()
            elif _is_list_type(v):
                ret[k] = self.__class__(seperator=self.seperator, ordered=self.ordered).from_dict(dict(enumerate(v)))
                ret[k] = ret[k].to_dict()
                ret[k] = list(ret[k].values())
            else:
                ret[k] = v

        return ret

    def to_yaml(self, *args, **kwargs):
        """Return YAML represenation of self."""
        return yaml.dump(self.to_dict(), *args, **kwargs)

    def update(self, other=(), **kwargs):  # pylint: disable=arguments-differ
        """Update (like dict().update), but accept dict_of_dicts as input."""

        if other and isinstance(other, Mapping):
            for k, v in other.items():
                if _is_dict_type(v):
                    self[k] = self.get(k, self.__class__(seperator=self.seperator, ordered=self.ordered)).update(v)
                elif _is_list_type(v):
                    self[k] = self.__class__(seperator=self.seperator, ordered=self.ordered).from_dict(
                        dict(enumerate(v)),
                    )
                    self[k].update(dict(enumerate(v)))
                    self[k] = list(self[k].values())
                else:
                    self[k] = v

        elif other and isinstance(other, Collection):
            raise NotImplementedError
            for (k, v) in other:
                if _is_dict_type(v):
                    self[k] = self.get(k, self.__class__(seperator=self.seperator, ordered=self.ordered)).update(v)
                else:
                    self[k] = v

        for k, v in kwargs.items():
            raise NotImplementedError
            if _is_dict_type(v):
                self[k] = self.get(k, self.__class__(seperator=self.seperator, ordered=self.ordered)).update(v)
            else:
                self[k] = v

        return self

    def apply(self, func, copy=False, key_filter=None, contains_filter=None, apply_to_lists=False):
        """Apply a function of values stored in self, optionally filtering or doing a deep copy."""
        # TODO apply_to_lists is a nasty hack, really need a !distribution type decorator in the yaml
        ret = deepcopy(self) if copy else self

        for k, v in ret.items():
            if _is_dict_type(v):  # isinstance(v, type(ret)):
                if contains_filter is not None:
                    key_set = set((contains_filter,) if isinstance(contains_filter, str) else contains_filter)
                    if key_set.issubset(v.keys()):
                        ret[k] = func(v)
                        continue
                ret[k] = v.apply(func, copy=copy, key_filter=key_filter, contains_filter=contains_filter)
            elif _is_list_type(v) and not apply_to_lists:
                ret[k] = self.__class__(seperator=self.seperator, ordered=self.ordered).from_dict(dict(enumerate(v)))
                if contains_filter is not None:
                    key_set = set((contains_filter,) if isinstance(contains_filter, str) else contains_filter)
                    if key_set.issubset(dict(enumerate(v)).keys()):
                        ret[k] = func(v)
                        ret[k] = list(ret[k].values())
                        continue
                ret[k] = ret[k].apply(func, copy=copy, key_filter=key_filter, contains_filter=contains_filter)
                ret[k] = list(ret[k].values())
            else:
                if key_filter is not None:
                    if k == key_filter:
                        ret[k] = func(v)
                elif contains_filter is not None:
                    ret[k] = v
                else:
                    ret[k] = func(v)

        return ret


if __name__ == "__main__":

    test_dict = {"a": "a", "b": {"c": "c", "d": "d"}}
    print(test_dict)  # noqa: T201

    nd = NestedDict(test_dict)
    print(nd)  # noqa: T201

    up = {"b": {"d": 12}}
    print(nd.update(up))  # noqa: T201

    nd.apply(lambda x: x + "a" if isinstance(x, str) else x)
    print(nd)  # noqa: T201
