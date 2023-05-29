import distance
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

def calculate_edit_distance():
    dataset = args.dataset
    if args.with_fonts:
        dataset += "_with_fonts"
    if args.with_boldface:
        dataset += "_boldface"
    tgt = open(f"logs/{dataset}_final_targets.txt").readlines()
    pred = open(f"logs/{dataset}_final_preds.txt").readlines()

    total_lev_dist = 0
    total_length = 0

    for t, p in zip(tgt, pred):
        t, p = t.split(), p.split()
        lev_dist = distance.levenshtein(t, p)
        total_lev_dist += lev_dist
        total_length += max(len(t), len(p))

    return total_lev_dist / total_length


if __name__ == "__main__":
    print("edit distance:  ", calculate_edit_distance())
