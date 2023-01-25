import acsets, petris

sir = petris.Petri()
s,i,r = sir.add_species(3)
inf,rec = sir.add_transitions([([s,i],[i,i]),([i],[r])])

serialized = sir.write_json()
deserialized = petris.Petri.read_json(petris.SchPetri, serialized)
reserialized = deserialized.write_json()

print(serialized)
print(reserialized)
