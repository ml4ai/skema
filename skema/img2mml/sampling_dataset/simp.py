from skema.img2mml.preprocessing.preprocess_mml import simplification
import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--idx")
args = parser.parse_args()

i = args.idx

open(f"sm_{i}.txt", "w").write
(
    simplification
    (
        open(f"smr_{i}.txt").readlines()[0]
    )
)
