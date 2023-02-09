import acsets, petris

s = ['s', 'i', 'r']

t = ['beta', 'gamma']

sir = petris.Petri()
s,i,r = sir.add_species(3)
inf,rec = sir.add_transitions([([s,i],[i,i]),([i],[r])])


for i, tran in enumerate(t): 
    sir.set_subpart(i, petris.attr_tname, t[i])

for j, spec in enumerate(s): 
    sir.set_subpart(j, petris.attr_sname, s[j])

serialized = sir.write_json()
deserialized = petris.Petri.read_json(petris.SchPetri, serialized)
reserialized = deserialized.write_json()

print(serialized)
print(reserialized)
