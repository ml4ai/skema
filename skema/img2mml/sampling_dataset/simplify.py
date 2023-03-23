from preprocessing.preprocess_mml import simplification
import argparse

# opening config file
parser = argparse.ArgumentParser()
parser.add_argument("--mml", required=True)
args = parser.parse_args()

f=open(args.config).readlines()[0]
simp_mml = simplification(f)
