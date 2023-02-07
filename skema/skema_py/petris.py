from acsets import Ob, Hom, Attr, AttrType, Schema, ACSet

from typing import Tuple, List

Species = Ob("S")
Transition = Ob("T")
Input = Ob("I")
Output = Ob("O")

hom_it = Hom("it", Input, Transition)
hom_is = Hom("is", Input, Species)
hom_ot = Hom("ot", Output, Transition)
hom_os = Hom("os", Output, Species)

Name = AttrType("Name", str)

attr_tname = Attr("tname", Transition, Name)
attr_sname = Attr("sname", Species, Name)

SchPetri = Schema(
    [Species, Transition, Input, Output],
    [hom_it, hom_is, hom_ot, hom_os],
    [Name],
    [attr_tname, attr_sname]
)

class Petri(ACSet):
    def __init__(self, schema=SchPetri):
        super(Petri, self).__init__(schema)

    def add_species(self, n: int) -> range:
        return self.add_parts(Species, n)

    def add_transitions(self, transitions: List[Tuple[List[int], List[int]]]) -> range:
        ts = self.add_parts(Transition, len(transitions))
        for (t, (ins, outs)) in zip(ts, transitions):
            for s in ins:
                arc = self.add_part(Input)
                self.set_subpart(arc, hom_it, t)
                self.set_subpart(arc, hom_is, s)
            for t in ins:
                arc = self.add_part(Output)
                self.set_subpart(arc, hom_ot, t)
                self.set_subpart(arc, hom_os, s)
        return ts
