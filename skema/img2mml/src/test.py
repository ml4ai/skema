# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from skema.img2mml.utils.utils import *


def evaluate(
    model,
    model_type,
    batch_size,
    test_dataloader,
    criterion,
    device,
    vocab,
    beam_params=None,
    is_test=False,
    ddp=False,
    rank=None,
    g2p=False,
):

    model.eval()
    epoch_loss = 0

    if is_test:
        mml_seqs = open("logs/test_targets_100K.txt", "w")
        pred_seqs = open(f"logs/test_predicted_100K.txt", "w")

    with torch.no_grad():

        torch.set_printoptions(profile="full")

        for i, (img, mml) in enumerate(test_dataloader):
            batch_size = mml.shape[0]
            mml = mml.to(device, dtype=torch.long)
            img = img.to(device)
            # imgs = list()
            # for im in img:
            #     imgs.append(torch.load(f"training_data/sample_data/image_tensors/{int(im.item())}.txt"))
            # img = torch.stack(imgs)

            """
            we will pass "mml" just to provide initial <sos> token.
            There will no teacher forcing while validation and testing.
            """
            outputs, preds = model(
                img, mml, is_test=is_test
            )  # O: (B, max_len, output_dim), preds: (B, max_len)

            if is_test:
                preds = garbage2pad(preds, vocab, is_test=is_test)
                output_dim = outputs.shape[-1]
                mml_reshaped = mml[:, 1:].contiguous().view(-1)
                if model_type == "opennmt":
                    outputs_reshaped = (
                        outputs[:, 1:, :].contiguous().view(-1, output_dim)
                    )  # (B * max_len-1, output_dim)
                elif model_type == "cnn_xfmer" or model_type == "resnet_xfmer":
                    outputs_reshaped = outputs.contiguous().view(
                        -1, output_dim
                    )  # (B * max_len-1, output_dim)

            else:
                output_dim = outputs.shape[-1]
                if model_type == "opennmt":
                    outputs_reshaped = (
                        outputs[:, 1:, :].contiguous().view(-1, output_dim)
                    )  # (B * max_len-1, output_dim)
                elif model_type == "cnn_xfmer" or model_type == "resnet_xfmer":
                    outputs_reshaped = outputs.contiguous().view(
                        -1, output_dim
                    )  # (B * max_len-1, output_dim)
                mml_reshaped = mml[:, 1:].contiguous().view(-1)

            loss = criterion(outputs_reshaped, mml_reshaped)

            epoch_loss += loss.item()

            if is_test:
                for idx in range(batch_size):
                    # writing target eqn
                    mml_arr = [vocab.itos[imml] for imml in mml[idx, :]]
                    mml_seq = " ".join(mml_arr)
                    mml_seqs.write(mml_seq + "\n")

                    # writing pred eqn
                    if beam_params is not None:
                        (
                            beam_k,
                            alpha,
                            min_length_bean_search_normalization,
                        ) = beam_params
                        predicted_seq = beam_search(
                            outputs[idx, :, :],
                            beam_k,
                            alpha,
                            min_length_bean_search_normalization,
                        )  # list of all eqns and score
                        pred = predicted_seq[0][0]

                    pred_arr = [
                        vocab.itos[ipred] for ipred in preds.int()[idx, :]
                    ]
                    pred_seq = " ".join(pred_arr)
                    pred_seqs.write(pred_seq + "\n")

    net_loss = epoch_loss / len(test_dataloader)
    return net_loss
