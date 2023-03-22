from preprocessing.preprocess_mml import simplification
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mml")

args = parser.parse_args()

simp_mml = simplification(args.mml)
