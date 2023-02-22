# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from utils.utils import *

def train(model, model_type, train_dataloader, vocab, optimizer, clip, ddp=False, rank=None):

    # train mode is ON i.e. dropout and normalization tech. will be used
    model.train()

    epoch_loss = 0

    # for i, (img, mml) in enumerate(train_dataloader):
    for i, (img, mml) in enumerate(train_dataloader):
        # mml: (B, max_len)
        # img: (B, in_channel, H, W)
        # print("mml shape: ", mml.shape)
        # print("img shape: ", img.shape)

        batch_size = mml.shape[0]
        if ddp:
            mml = mml.to(f"cuda:{rank}", dtype=torch.long)
            img = img.to(f"cuda:{rank}")
        else:
            mml = mml.to("cuda", dtype=torch.long)
            img = img.to("cuda")

        # setting gradients to zero
        optimizer.zero_grad()

        outputs, _ = model(img, mml) #(B, max_len, output_dim)
        # print("shape:", (mml.shape, outputs.shape))

        # if masking:
        #     # output: (l*B, output_dim); mml: (l*B)
        #     outputs, mml = masking_pad_token(outputs, mml)
        # else:
        output_dim = outputs.shape[-1]
        # avoiding <sos> token while Calculating loss
        mml = mml[:,1:].contiguous().view(-1)
        if model_type=="opennmt":
            outputs = outputs[:,1:,:].contiguous().view(-1, output_dim)
        elif model_type=="cnn_xfmer":
            outputs = outputs.contiguous().view(-1, output_dim)

        # print("outputs mml devices are: ", outputs.get_device(), mml.get_device())
        # print("shape:", (mml.shape, outputs.shape))
        loss = calculate_loss(outputs, mml, vocab)
        # print("loss devices are: ", loss.get_device())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    net_loss = epoch_loss/len(train_dataloader)
    return net_loss
