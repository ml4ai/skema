import os


def recover_mml(eqn):
    _arr = eqn.split()
    _idx = 0
    _temp = list()
    while _idx < len(_arr) - 1:
        if _arr[_idx + 1] in ["__<mi>", "__<mo>", "__<mn>"]:
            begin_mml_tok = _arr[_idx + 1][2:]
            end_mml_tok = begin_mml_tok[0] + "/" + begin_mml_tok[1:]
            _temp.append(begin_mml_tok)
            _temp.append(_arr[_idx])
            _temp.append(end_mml_tok)
            _idx += 2

        else:
            _temp.append(_arr[_idx])
            _idx += 1

    return " ".join(_temp) + f" {_arr[-1]}"


if __name__ == "__main__":
    tgt_file = open("logs/test_targets_100K.txt").readlines()
    pred_file = open("logs/test_predicted_100K.txt").readlines()
    tgt_new = open("logs/test_targets_100K_recovered.txt", "w")
    pred_new = open("logs/test_predicted_100K_recovered.txt", "w")

    for (t, p) in zip(tgt_file, pred_file):
        _t, _p = map(recover_mml, (t, p))
        tgt_new.write(_t)
        pred_new.write(_p)

    os.remove("logs/test_targets_100K.txt")
    os.remove("logs/test_predicted_100K.txt")
    os.rename(
        "logs/test_targets_100K_recovered.txt", "logs/test_targets_100K.txt"
    )
    os.rename(
        "logs/test_predicted_100K_recovered.txt",
        "logs/test_predicted_100K.txt",
    )
