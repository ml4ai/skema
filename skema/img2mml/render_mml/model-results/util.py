import json


def get_bins():
    return [i for i in range(0, 400, 50)]


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data
