import distance


def calculate_edit_distance():
    tgt = open("logs/final_targets.txt").readlines()
    pred = open("logs/final_preds.txt").readlines()

    total_lev_dist = 0
    total_length = 0

    for (t, p) in zip(tgt, pred):
        t, p = t.split(), p.split()
        lev_dist = distance.levenshtein(t, p)
        total_lev_dist += lev_dist
        total_length += max(len(t), len(p))

    return total_lev_dist / total_length


if __name__ == "__main__":
    print("edit distance:  ", calculate_edit_distance())
