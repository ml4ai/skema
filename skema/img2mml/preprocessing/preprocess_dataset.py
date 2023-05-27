# creating tab dataset having image_num and mml
# split train, test, val
# dataloader to load data
import pandas as pd
import random
import torch
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from collections import Counter
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import SequentialSampler
from functools import partial
from skema.img2mml.utils.utils import CreateVocab
import argparse
import json
import pickle

# opening config file
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
parser.add_argument(
    "--config",
    help="configuration file for paths and hyperparameters",
    default="configs/xfmer_mml_config.json",
)

args = parser.parse_args()

dataset = args.dataset
if args.with_fonts:
    dataset += "_with_fonts"

with open(args.config, "r") as cfg:
    config = json.load(cfg)
    config["dataset"] = dataset
    if args.with_boldface:
        config["with_boldface"] = "True"
    else:
        config["with_boldface"] = "False"


class Img2MML_dataset(Dataset):
    def __init__(self, dataframe, vocab, tokenizer):
        self.dataframe = dataframe
        self.vocab = vocab

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        eqn = self.dataframe.iloc[index, 1]
        indexed_eqn = []
        for token in eqn.split():
            if self.vocab.stoi[token] != None:
                indexed_eqn.append(self.vocab.stoi[token])
            else:
                indexed_eqn.append(self.vocab.stoi["<unk>"])

        return self.dataframe.iloc[index, 0], torch.Tensor(indexed_eqn)


class My_pad_collate(object):
    """
    padding mml to max_len, and stacking images
    return: mml_tensors of shape [batch, max_len]
            stacked image_tensors [batch]
    """

    def __init__(self, device, vocab, max_len):
        self.device = device
        self.vocab = vocab
        self.max_len = max_len
        self.pad_idx = vocab.stoi["<pad>"]

    def __call__(self, batch):
        _img, _mml = zip(*batch)

        # padding mml
        # padding to a fix max_len equations with more tokens than
        # max_len will be chopped down to max_length.

        batch_size = len(_mml)
        padded_mml_tensors = (
            torch.ones([batch_size, self.max_len], dtype=torch.long) * self.pad_idx
        )
        for b in range(batch_size):
            if len(_mml[b]) <= self.max_len:
                padded_mml_tensors[b][: len(_mml[b])] = _mml[b]
            else:
                padded_mml_tensors[b][: self.max_len] = _mml[b][: self.max_len]

        # images tensors
        _img = torch.Tensor(_img)

        return (
            _img.to(self.device),
            padded_mml_tensors.to(self.device),
        )


def main():
    print("preprocessing data...")

    # reading raw text files
    if config["with_boldface"] == "True":
        mml_path = f"{config['data_path']}/{config['dataset_type']}/{config['dataset']}/{config['markup']}.lst"
    else:
        mml_path = f"{config['data_path']}/{config['dataset_type']}/{config['dataset']}/{config['markup']}_boldface.lst"

    img_tnsr_path = f"{config['data_path']}/{config['dataset_type']}/{config['dataset']}/image_tensors"
    mml_txt = open(mml_path).read().split("\n")[:-1]
    image_num = range(0, len(mml_txt))

    # split the image_num into train, test, validate
    train_val_images, test_images = train_test_split(
        image_num, test_size=0.1, random_state=42
    )
    train_images, val_images = train_test_split(
        train_val_images, test_size=0.1, random_state=42
    )

    for t_idx, t_images in enumerate([train_images, test_images, val_images]):
        raw_mml_data = {
            "IMG": [num for num in t_images],
            "EQUATION": [("<sos> " + mml_txt[num] + " <eos>") for num in t_images],
        }

        if t_idx == 0:
            train = pd.DataFrame(raw_mml_data, columns=["IMG", "EQUATION"])
        elif t_idx == 1:
            test = pd.DataFrame(raw_mml_data, columns=["IMG", "EQUATION"])
        else:
            val = pd.DataFrame(raw_mml_data, columns=["IMG", "EQUATION"])

    # build vocab
    print("building vocab...")

    counter = Counter()
    for line in train["EQUATION"]:
        counter.update(line.split())

    # <unk>, <pad> will be prepended in the vocab file
    vocab = Vocab(
        counter,
        min_freq=config["vocab_freq"],
        specials=["<pad>", "<unk>", "<sos>", "<eos>"],
    )

    # writing vocab file...
    dataset = config["dataset"]
    if config["with_boldface"] == "True":
        vfile = open(
            f"{config['data_path']}/sample_data/{config['dataset']}/{dataset}_with_boldface_vocab.txt",
            "w",
        )
    else:
        vfile = open(
            f"{config['data_path']}/sample_data/{config['dataset']}/{dataset}_vocab.txt",
            "w",
        )
    for vidx, vstr in vocab.stoi.items():
        vfile.write(f"{vidx} \t {vstr} \n")

    # define tokenizer function
    tokenizer = lambda x: x.split()

    print("saving dataset files to data/ folder...")

    if config["with_boldface"] == "True":
        train.to_csv(
            f"{config['data_path']}/sample_data/{config['dataset']}/train_bold.csv",
            index=False,
        )
        test.to_csv(
            f"{config['data_path']}/sample_data/{config['dataset']}/test_bold.csv",
            index=False,
        )
        val.to_csv(
            f"{config['data_path']}/sample_data/{config['dataset']}/val_bold.csv",
            index=False,
        )
    else:
        train.to_csv(
            f"{config['data_path']}/sample_data/{config['dataset']}/train.csv",
            index=False,
        )
        test.to_csv(
            f"{config['data_path']}/sample_data/{config['dataset']}/test.csv",
            index=False,
        )
        val.to_csv(
            f"{config['data_path']}/sample_data/{config['dataset']}/val.csv",
            index=False,
        )

    print("building dataloaders...")

    # initializing pad collate class
    mypadcollate = My_pad_collate(config["device"], vocab, config["max_len"])

    # initailizing class Img2MML_dataset: train dataloader
    imml_train = Img2MML_dataset(train, vocab, tokenizer)
    # creating dataloader
    if config["DDP"]:
        train_sampler = DistributedSampler(
            dataset=imml_train,
            num_replicas=config["num_DDP_gpus"],
            rank=config["rank"],
            shuffle=True,
        )
        sampler = train_sampler
        shuffle = False
    else:
        sampler = None
        shuffle = config["shuffle"]

    train_dataloader = DataLoader(
        imml_train,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=mypadcollate,
        pin_memory=config["pin_memory"],
    )
    if config["with_boldface"] == "True":
        train_dl = f"{config['data_path']}/sample_data/{config['dataset']}/train_bold_dataloader.pkl"
    else:
        train_dl = f"{config['data_path']}/sample_data/{config['dataset']}/train_dataloader.pkl"
    with open(train_dl, "wb") as file:
        pickle.dump(train_dataloader, file)

    # initailizing class Img2MML_dataset: test dataloader
    imml_test = Img2MML_dataset(test, vocab, tokenizer)

    if config["DDP"]:
        test_sampler = SequentialSampler(imml_test)
        sampler = test_sampler
        shuffle = False
    else:
        sampler = None
        shuffle = config["shuffle"]

    test_dataloader = DataLoader(
        imml_test,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=mypadcollate,
        pin_memory=config["pin_memory"],
    )
    if config["with_boldface"] == "True":
        test_dl = f"{config['data_path']}/sample_data/{config['dataset']}/test_bold_dataloader.pkl"
    else:
        test_dl = (
            f"{config['data_path']}/sample_data/{config['dataset']}/test_dataloader.pkl"
        )
    with open(test_dl, "wb") as file:
        pickle.dump(test_dataloader, file)

    # initailizing class Img2MML_dataset: val dataloader
    imml_val = Img2MML_dataset(val, vocab, tokenizer)

    if config["DDP"]:
        val_sampler = SequentialSampler(imml_val)
        sampler = val_sampler
        shuffle = False
    else:
        sampler = None
        shuffle = config["shuffle"]

    val_dataloader = DataLoader(
        imml_val,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=mypadcollate,
        pin_memory=config["pin_memory"],
    )
    if config["with_boldface"] == "True":
        val_dl = f"{config['data_path']}/sample_data/{config['dataset']}/val_bold_dataloader.pkl"
    else:
        val_dl = (
            f"{config['data_path']}/sample_data/{config['dataset']}/val_dataloader.pkl"
        )
    with open(val_dl, "wb") as file:
        pickle.dump(val_dataloader, file)

    if config["with_boldface"] == "True":
        voc_data = (
            f"{config['data_path']}/sample_data/{config['dataset']}/voc_bold_data.pkl"
        )
    else:
        voc_data = f"{config['data_path']}/sample_data/{config['dataset']}/voc_data.pkl"

    with open(voc_data, "wb") as file:
        pickle.dump(vocab, file)

    print("Dataset is ready to train.")


if __name__ == "__main__":
    main()