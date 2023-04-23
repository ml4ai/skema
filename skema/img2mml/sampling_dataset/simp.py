from skema.img2mml.preprocessing.preprocess_mml import simplification
import os

smr = open("smr.txt").readlines()[0]
sm  = open("sm.txt", "w")

sm.write(simplification(smr))
