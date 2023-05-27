# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from skema.img2mml.utils.utils import *


def train(
    model,
    model_type,
    img_tnsr_path,
    train_dataloader,
    optimizer,
    criterion,
    clip,
    device,
    ddp=False,
    rank=None,
):
    # train mode is ON i.e. dropout and normalization tech. will be used
    model.train()

    epoch_loss = 0

    for i, (img, mml) in enumerate(train_dataloader):
        # mml: (B, max_len)
        # img: (B, in_channel, H, W)
        batch_size = mml.shape[0]
        mml = mml.to(device, dtype=torch.long)
        imgs = list()
        for im in img:
            imgs.append(torch.load(f"{img_tnsr_path}/{int(im.item())}.txt"))
        img = torch.stack(imgs).to(device)

        # setting gradients to zero
        optimizer.zero_grad()

        outputs, _ = model(img, mml)  # (B, max_len, output_dim)
        output_dim = outputs.shape[-1]

        # avoiding <sos> token while Calculating loss
        mml = mml[:, 1:].contiguous().view(-1)
        if model_type == "opennmt":
            outputs = outputs[:, 1:, :].contiguous().view(-1, output_dim)
        elif model_type == "cnn_xfmer" or model_type == "resnet_xfmer":
            outputs = outputs.contiguous().view(-1, output_dim)

        loss = criterion(outputs, mml)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    net_loss = epoch_loss / len(train_dataloader)
    return net_loss
