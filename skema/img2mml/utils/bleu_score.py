import os
from torchtext.data.metrics import bleu_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    choices=["arxiv", "im2mml", "arxiv_im2mml"],
    default="arxiv_im2mml",
    help="Choose which dataset to be used for training. Choices: arxiv, im2mml, arxiv_im2mml.",
)
parser.add_argument(
    "--with_fonts",
    action="store_true",
    default=False,
    help="Whether using the dataset with diverse fonts",
)
parser.add_argument(
    "--with_boldface",
    action="store_true",
    default=False,
    help="Whether having boldface in labels",
)
args = parser.parse_args()


def calculate_bleu_score():
    dataset = args.dataset
    if args.with_fonts:
        dataset += "_with_fonts"
    if args.with_boldface:
        dataset += "_boldface"

    tt = open(f"logs/test_{dataset}_targets_100K.txt").readlines()
    tp = open(f"logs/test_{dataset}_predicted_100K.txt").readlines()
    _tt = open(f"logs/{dataset}_final_targets.txt", "w")
    _tp = open(f"logs/{dataset}_final_preds.txt", "w")

    for i, j in zip(tt, tp):
        eos_i = i.find("<eos>")
        _tt.write(i[6:eos_i] + "\n")

        eos_j = j.find("<eos>")
        _tp.write(j[6:eos_j] + "\n")

    test = open(f"logs/{dataset}_final_targets.txt").readlines()
    predicted = open(f"logs/{dataset}_final_preds.txt").readlines()

    candidate_corpus, references_corpus = [], []

    for t, p in zip(test, predicted):
        candidate_corpus.append(t.split())
        references_corpus.append([p.split()])

    bleu = bleu_score(candidate_corpus, references_corpus)
    return bleu


if __name__ == "__main__":
    print(" calculating Bleu Score...  ")
    tt_bleu = calculate_bleu_score()
    print(" torchtext BLEU score: ", tt_bleu)
