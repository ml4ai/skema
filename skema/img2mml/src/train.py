# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from skema.img2mml.utils.utils import *
import re
from torchtext.vocab import Vocab


def check_mathml_syntax(
    outputs: torch.Tensor, mml: torch.Tensor, vocab: Vocab
) -> float:
    """
    Check the syntax of MathML expressions in outputs and mml.

    Args:
        outputs (torch.Tensor): Predicted MathML expressions. Shape: (batch_size, max_token_length, output_dim).
        mml (torch.Tensor): Ground truth MathML expressions. Shape: (batch_size, max_token_length).
        vocab (torchtext.vocab.Vocab): Vocabulary mapping token indices to tokens.

    Returns:
        float: Accuracy of MathML syntax, i.e., the percentage of correct MathML expressions.

    Note:
        This function checks the usage of tag pairs, nested structure, and tag attributes in MathML expressions.
        It assumes that the start and end tags (<math> and </math>) are correctly used in both outputs and mml.

    """
    outputs = outputs
    batch_size = outputs.size(0)
    output_dim = outputs.size(2)
    correct_count = 0

    for i in range(batch_size):
        output_tokens = [
            vocab.itos[idx] for idx in torch.argmax(F.softmax(outputs[i], dim=1), dim=1)
        ]
        mml_tokens = [vocab.itos[idx] for idx in mml[i]]

        output_str = " ".join(output_tokens)
        mml_str = " ".join(mml_tokens)

        # Remove <EOS> and everything after it
        output_str = output_str.split(" <eos>", 1)[0]
        mml_str = mml_str.split(" <eos>", 1)[0]
        # Check the start and end tags
        if output_str.startswith("<math") and output_str.endswith("</math>"):
            # Extract tag pairs
            output_tags = re.findall(r"<[^>]+>", output_str)

            # Check tag pair usage and nested structure
            tag_stack = []
            for o_tag in output_tags:
                if "mspace" not in o_tag:
                    o_tag_name = o_tag.split(" ", 1)[0]

                    if "/" not in o_tag:
                        tag_stack.append(o_tag_name.replace("<", "").replace(">", ""))
                    else:
                        if tag_stack and tag_stack[-1] == o_tag_name.replace(
                            "<", ""
                        ).replace(">", "").replace("/", ""):
                            tag_stack.pop()
                        else:
                            break

            if not tag_stack:
                correct_count += 1

    accuracy = correct_count / batch_size
    return accuracy


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
    vocab=None
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
        syntax_acc = check_mathml_syntax(outputs, mml[:, 1:], vocab)
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
