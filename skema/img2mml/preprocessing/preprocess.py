# creating tab dataset having image_num and mml
# split train, test, val
# dataloader to load data
import numpy as np
import pandas as pd
import random
import torch
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from collections import Counter
# from torchtext.legacy.vocab import Vocab
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import SequentialSampler
from functools import partial
from utils.utils import CreateVocab


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
                indexed_eqn.append(self.vocab.stoi['<unk>'])

        return self.dataframe.iloc[index, 0],torch.Tensor(indexed_eqn)


class My_pad_collate(object):
    """
    padding mml to max_len, and stacking images
    return: mml_tensors of shape [batch, max_len]
            stacked image_tensors [batch]
    """
    def __init__(self, device, vocab,
                    max_len):
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
        padded_mml_tensors = torch.ones([batch_size, self.max_len], dtype=torch.long) * self.pad_idx
        for b in range(batch_size):
            if len(_mml[b]) <= self.max_len:
                padded_mml_tensors[b][:len(_mml[b])] = _mml[b]
            else:
                padded_mml_tensors[b][:self.max_len] = _mml[b][:self.max_len]

        # images tensors
        _img = [i for i in _img]

        return torch.stack(_img).to(self.device), padded_mml_tensors.to(self.device)


def preprocess_mml(config):

    print('preprocessing data...')

    # reading raw text files
    MMLPath = f"{config['data_path']}/{config['dataset_type']}/{config['mml_path']}"
    IMGTnsrPath = f"{config['data_path']}/{config['dataset_type']}/{config['image_path']}"
    mml_txt = open(MMLPath).read().split('\n')[:-1]
    # image_num = range(0,len(mml_txt))
    # raw_mml_data = {'IMG': [torch.load(f'{IMGTnsrPath}/{num}.txt') for num in image_num],
    #                 'EQUATION': [('<sos> '+ mml + ' <eos>') for mml in mml_txt]}

    # adding <sos> and <eos> tokens then creating a dataframe
    equation_array = list()
    image_array = list()
    for num in range(len(mml_txt)):
        if "REJECTED MATHML" not in mml_txt[num]:
            # appending mmls
            mml = '<sos> '+ mml_txt[num] + ' <eos>'
            equation_array.append(mml)
            # appending image tensors
            image_array.append(torch.load(f'{IMGTnsrPath}/{num}.txt'))

    raw_mml_data = {'IMG': image_array,'EQUATION': equation_array}

    df = pd.DataFrame(raw_mml_data, columns=['IMG','EQUATION'])

    train_val, test = train_test_split(df, test_size = 0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1, random_state=42)

    # build vocab
    print("building vocab...")

    counter = Counter()
    for line in train['EQUATION']:
        counter.update(line.split())

    # <unk>, <pad> will be prepended in the vocab file
    vocab = Vocab(counter, min_freq=config["vocab_freq"], specials=['<pad>', '<unk>', '<sos>', '<eos>'])

    # special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]
    # vocab = CreateVocab(train, special_tokens, min_freq=10)
    # print("vocab size: ", vocab.__len__())

    # writing vocab file...
    vfile = open('vocab.txt', 'w')
    for vidx, vstr in vocab.stoi.items():
        vfile.write(f'{vidx} \t {vstr} \n')

    # define tokenizer function
    tokenizer = lambda x: x.split()

    print("saving dataset files to data/ folder...")

    train.to_csv(f"{config['data_path']}/{config['dataset_type']}/train.csv", index=False)
    test.to_csv(f"{config['data_path']}/{config['dataset_type']}/test.csv", index=False)
    val.to_csv(f"{config['data_path']}/{config['dataset_type']}/val.csv", index=False)


    print("building dataloaders...")

    # initializing pad collate class
    mypadcollate = My_pad_collate(config["device"], vocab,
                                    config["max_len"])

    # initailizing class Img2MML_dataset: train dataloader
    imml_train = Img2MML_dataset(train,
                                 vocab,
                                 tokenizer)
    # creating dataloader
    if config["DDP"]:
        train_sampler = DistributedSampler(dataset=imml_train,
                                        num_replicas=config["world_size"],
                                        rank=config["rank"],
                                        shuffle=True)
        sampler = train_sampler
        shuffle = False
    else:
        sampler=None
        shuffle = config["shuffle"]
    train_dataloader = DataLoader(imml_train,
                                  batch_size=config["batch_size"],
                                  num_workers=config["num_workers"],
                                  shuffle=shuffle,
                                  sampler=sampler,
                                  collate_fn=mypadcollate,
                                  pin_memory=config["pin_memory"])

    # initailizing class Img2MML_dataset: test dataloader
    imml_test = Img2MML_dataset(test,
                                vocab,
                                tokenizer)

    if config["DDP"]:
        # test_sampler = DistributedSampler(dataset=imml_test,
        #                                 num_replicas=config["world_size"],
        #                                 rank=config["rank"],
        #                                 shuffle=True)
        test_sampler = SequentialSampler(imml_test)
        sampler = test_sampler
        shuffle = False
    else:
        sampler=None
        shuffle = config["shuffle"]

    test_dataloader = DataLoader(imml_test,
                                 batch_size=config["batch_size"],
                                 num_workers=config["num_workers"],
                                 shuffle=shuffle,
                                 sampler=sampler,
                                 collate_fn=mypadcollate,
                                 pin_memory=config["pin_memory"])

    # initailizing class Img2MML_dataset: val dataloader
    imml_val = Img2MML_dataset(val,
                               vocab,
                               tokenizer)

    if config["DDP"]:
        # val_sampler = DistributedSampler(dataset=imml_val,
        #                                 num_replicas=config["world_size"],
        #                                 rank=config["rank"],
        #                                 shuffle=True)
        val_sampler = SequentialSampler(imml_val)
        sampler = val_sampler
        shuffle = False
    else:
        sampler=None
        shuffle = config["shuffle"]

    val_dataloader = DataLoader(imml_val,
                                batch_size=config["batch_size"],
                                num_workers=config["num_workers"],
                                shuffle=shuffle,
                                sampler=sampler,
                                collate_fn=mypadcollate,
                                pin_memory=config["pin_memory"])

    return train_dataloader, test_dataloader, val_dataloader, vocab
