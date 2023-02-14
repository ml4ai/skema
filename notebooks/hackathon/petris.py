from acsets import Ob, Hom, Attr, AttrType, Schema, ACSet

Species = Ob("S")
Transition = Ob("T")
Input = Ob("I")
Output = Ob("O")

hom_it = Hom("it", Input, Transition)
hom_is = Hom("is", Input, Species)
hom_ot = Hom("ot", Output, Transition)
hom_os = Hom("os", Output, Species)

Name = AttrType("Name", str)
Uid = AttrType("uid", int)

attr_tuid = Attr("uid", Transition, Uid)
attr_suid = Attr("uid", Species, Uid)

attr_tname = Attr("tname", Transition, Name)
attr_sname = Attr("sname", Species, Name)

SchPetri = Schema(
    [Species, Transition, Input, Output],
    [hom_it, hom_is, hom_ot, hom_os],
    [Name, Uid],
    [attr_tname, attr_sname, attr_tuid, attr_suid]
)

class Petri(ACSet):
    def __init__(self, schema=SchPetri):
        super(Petri, self).__init__(schema)

    def add_species(self, n: int) -> range:
        return self.add_parts(Species, n)

    def add_transitions(self, transitions: list[tuple[list[int], list[int]]]) -> range:
        ts = self.add_parts(Transition, len(transitions))
        for (t, (ins, outs)) in zip(ts, transitions):
            for s in ins:
                arc = self.add_part(Input)
                self.set_subpart(arc, hom_it, t)
                self.set_subpart(arc, hom_is, s)
            for tt in outs:
                arc = self.add_part(Output)
                self.set_subpart(arc, hom_ot, t)
                self.set_subpart(arc, hom_os, tt)
        return ts
