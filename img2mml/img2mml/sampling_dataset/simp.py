from img2mml.preprocessing.preprocess_mml import simplification
import os, sys

i = sys.argv[-1]

sm = open(f"{os.getcwd()}/temp_folder/sm_{i}.txt", "w")
smr = (
    open(f"{os.getcwd()}/temp_folder/smr_{i}.txt")
    .readlines()[0]
    .strip()
)

sm.write(simplification(smr))
