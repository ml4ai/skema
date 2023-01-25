from dataclasses import dataclass
import json

@dataclass(eq=True, frozen=True)
class Ob:
    name: str

class Property:
    pass

@dataclass(eq=True, frozen=True)
class Hom(Property):
    name: str
    dom: Ob
    hom: Ob

@dataclass(eq=True, frozen=True)
class AttrType:
    name: str
    ty: type

@dataclass(eq=True, frozen=True)
class Attr(Property):
    name: str
    dom: Ob
    codom: AttrType

@dataclass(eq=True, frozen=True)
class Schema:
    """Schema for an acset"""
    obs: list[Ob]
    homs: list[Hom]
    attrtypes: list[AttrType]
    attrs: list[Attr]

    def props_outof(self, ob: Ob) -> list[Property]:
        return filter(lambda f: f.dom == ob, self.homs + self.attrs)

    def from_string(self, s: str):
        x = next((x for x in self.obs if x.name == s), None)
        if x != None:
            return x
        x = next((x for x in self.homs if x.name == s), None)
        if x != None:
            return x
        x = next((x for x in self.attrtypes if x.name == s), None)
        if x != None:
            return x
        x = next((x for x in self.attrs if x.name == s), None)
        if x != None:
            return x

class ACSet:
    schema: Schema
    _parts: dict[Ob, int]
    _subparts: dict[Property, dict[int, any]]

    def __init__(self, schema: Schema):
        self.schema = schema
        self._parts = { ob:0 for ob in schema.obs }
        self._subparts = { f:{} for f in schema.homs + schema.attrs }

    def add_parts(self, ob: Ob, n: int) -> range:
        assert(ob in self.schema.obs)
        i = self._parts[ob]
        self._parts[ob] += n
        return range(i, i+n)

    def add_part(self, ob: Ob) -> int:
        return self.add_parts(ob, 1)[0]

    def _check_type(self, f: Property, x: any):
        if f in self.schema.homs:
            assert(isinstance(x, int))
        elif f in self.schema.attrs:
            assert(isinstance(x, f.codom.ty))
        else:
            raise(f"{f} not found in schema")

    def set_subpart(self, i: int, f: Property, x: any):
        self._check_type(f, x)
        self._subparts[f][i] = x

    def subpart(self, i: int, f: Property, oneindex=False):
        if oneindex and type(f) == Hom:
            return self._subparts[f][i] + 1
        else:
            return self._subparts[f][i]

    def nparts(self, ob: Ob) -> int:
        assert(ob in self.schema.obs)
        return self._parts[ob]

    def parts(self, ob: Ob) -> range:
        return range(0, self.nparts(ob))

    def incident(self, x: any, f: Property) -> list[int]:
        self._check_type(f, x)
        return filter(lambda i: self.subpart(i, f) == x, self.parts(f.dom))

    def write_json(self):
        return json.dumps({
            ob.name: [self.prop_dict(ob, i) for i in self.parts(ob)] for ob in self.schema.obs
        })

    def prop_dict(self, ob: Ob, i: int) -> dict[str, any]:
        return { f.name: self.subpart(i, f, oneindex=True) for f in self.schema.props_outof(ob) }

    @classmethod
    def read_json(cls, schema: Schema, s: str):
        acs = cls(schema)
        d = json.loads(s)
        for (obname, proplist) in d.items():
            ob = schema.from_string(obname)
            for props in proplist:
                i = acs.add_part(ob)
                for (fname,v) in props.items():
                    f = schema.from_string(fname)
                    if type(f) == Hom:
                        acs.set_subpart(i, f, v-1)
                    else:
                        acs.set_subpart(i, f, v)
        return acs
