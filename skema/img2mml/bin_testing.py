import pandas as pd
import random
import torch
import os, sys
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from collections import Counter
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import SequentialSampler
from functools import partial

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
            torch.ones([batch_size, self.max_len], dtype=torch.long)
            * self.pad_idx
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


def bin_test_dataloader(config, vocab, start=None, end=None):

    # reading raw text files
    img_tnsr_path = f"{config['data_path']}/{config['dataset_type']}/image_tensors"
    df = pd.read_csv(f"{config['data_path']}/{config['dataset_type']}/test.csv")
    imgs, eqns = df["IMG"], df["EQUATION"]

    eqns_arr = list()
    imgs_arr = list()
    for i,e in zip(imgs, eqns):
        if start != None:
            if len(e.split()) > start and len(e.split()) <= end:
                eqns_arr.append(e)
                imgs_arr.append(i)
        else:
            eqns_arr.append(e)
            imgs_arr.append(i)


    raw_mml_data = {
        "IMG": imgs_arr,
        "EQUATION": eqns_arr,
    }
    test = pd.DataFrame(raw_mml_data, columns=["IMG", "EQUATION"])

    # define tokenizer function
    tokenizer = lambda x: x.split()

    # initializing pad collate class
    mypadcollate = My_pad_collate(device, vocab, max_len)

    # initailizing class Img2MML_dataset: test dataloader
    imml_test = Img2MML_dataset(test, vocab, tokenizer)
    test_dataloader = DataLoader(
        imml_test,
        batch_size=64,
        num_workers=0,
        shuffle=True,
        sampler=None,
        collate_fn=mypadcollate,
        pin_memory=False,
    )

    return test_dataloader
