# -*- coding: utf-8 -*-
import torch

# new addition
import math
import random

from tqdm.auto import tqdm
from skema.img2mml.utils.utils import *
from skema.img2mml.src.test import evaluate

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
    isBatchScheduler=False,
    reduce_on_plateau_scheduler=False,
    scheduler=None,
    val_dataloader=None, batch_size=None, vocab=None
):
    # train mode is ON i.e. dropout and normalization tech. will be used
    model.train()

    epoch_loss = 0
    tset = tqdm(iter(train_dataloader))
    latest_val_loss = 100

    # for i, (img, mml) in enumerate(train_dataloader):
    for i, (img, mml) in enumerate(tset):
        # mml: (B, max_len)
        # img: (B, in_channel, H, W)
        mml.shape[0]
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

        if (not ddp) or (ddp and rank == 0):
            desc = 'Loss: %.4f - Learning Rate: %.6f' % (loss.item(), optimizer.param_groups[0]['lr'])
            tset.set_description(desc)

        """
        new addition-------
        """
        if isBatchScheduler:
            # Calculating val_loss after every 1000 batches
            # of size 64.
            if i > 0 and i % 20 == 0 and i < len(train_dataloader):
                # randomly choosing 50 samples for the validation
                val_dataloader = random.sample(list(val_dataloader), 10)
                val_loss = evaluate(model,model_type,img_tnsr_path,
                                    batch_size,val_dataloader,criterion,
                                    device,vocab,ddp=ddp,rank=rank,)

                if val_loss < latest_val_loss:
                    latest_val_loss = val_loss
                    if (not ddp) or (ddp and rank == 0):
                        torch.save(
                            model.state_dict(),
                            f"trained_models/{model_type}_{dataset_type}_{config['markup']}_batch_best.pt",
                        )

                if reduce_on_plateau_scheduler:
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

                print(f"steps completed: {i} \t validation loss: {val_loss} \t validationn perplexity: {math.exp(val_loss):7.3f}")
            """
            ----------------------------------------------------------------------
            """
        else:
            net_loss = epoch_loss / len(train_dataloader)
            return net_loss
