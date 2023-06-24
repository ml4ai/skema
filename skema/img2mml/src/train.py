# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from skema.img2mml.utils.utils import *
import re
from torchtext.vocab import Vocab
import xml.etree.ElementTree as ET
import zss
from typing import List, Optional
import math
import torch.nn.functional as F


class MathMLNode:
    def __init__(
        self,
        tag: str,
        attributes: Optional[dict] = None,
        content: Optional[str] = None,
        children: Optional[List["MathMLNode"]] = None,
    ):
        """
        Represents a node in the MathML tree.

        Args:
            tag (str): The tag name of the MathML element.
            attributes (dict, optional): The attributes of the MathML element. Defaults to None.
            content (str, optional): The content of the MathML element. Defaults to None.
            children (List[MathMLNode], optional): The child nodes of the current node. Defaults to None.
        """
        self.tag = tag
        self.attributes = attributes if attributes else {}
        self.content = content
        self.children = children if children else []


def add_semicolon_to_unicode(string: str) -> str:
    """
    Checks if the string contains Unicode starting with '&#x' and adds a semicolon ';' after it.

    Args:
        string: The input string to check.

    Returns:
        The modified string with semicolons added after Unicode.
    """
    # Define a regular expression pattern to match '&#x' followed by hexadecimal characters
    pattern = r"&#x[0-9A-Fa-f]+"

    # Find all matches in the string using the pattern
    matches = re.findall(pattern, string)

    # Iterate over the matches and add semicolon after each Unicode
    for match in matches:
        string = string.replace(match, match + ";")

    return string


def convert_to_mathml_tree(mathml_string: str) -> MathMLNode:
    """
    Converts a MathML string into a MathML tree structure.

    Args:
        mathml_string (str): The MathML string to convert.

    Returns:
        MathMLNode: The root node of the MathML tree.
    """
    root = ET.fromstring(
        add_semicolon_to_unicode(mathml_string)
        .replace('<mspace linebreak="newline" />', "")
        .replace("<unk>", "unk")
    )
    return convert_to_mathml_node(root)


def convert_to_mathml_node(element: ET.Element) -> MathMLNode:
    """
    Recursively converts an XML element into a MathMLNode.

    Args:
        element (Element): The XML element to convert.

    Returns:
        MathMLNode: The converted MathMLNode.
    """
    attributes = dict(element.attrib)
    children = [convert_to_mathml_node(child) for child in element]
    return MathMLNode(element.tag, attributes, element.text.replace(" ", ""), children)


def calculate_tree_edit_distance(mathml_pred: str, mathml_label: str) -> int:
    """
    Calculates the tree edit distance between two MathML strings.

    Args:
        mathml_pred (str): The MathML prediction string.
        mathml_label (str): The MathML label string.

    Returns:
        int: The tree edit distance between the two MathML strings. If returning -1, it means input cannot be represented
         as a tree structure.
    """
    # If the token length difference is larger than 10, it returns -1
    if abs(mathml_pred.count(" ") - mathml_label.count(" ")) >= 10:
        return -1

    # If the prediction cannot make a tree structure, it returns -1.
    try:
        mathml_tree1 = convert_to_mathml_tree(mathml_pred)
    except:
        return -1
    # If the label cannot make a tree structure, it returns 0.
    try:
        mathml_tree2 = convert_to_mathml_tree(mathml_label)
    except:
        return 0

    def get_label(node: MathMLNode) -> str:
        """
        Retrieves the label for a MathMLNode.

        Args:
            node (MathMLNode): The MathMLNode to get the label for.

        Returns:
            str: The label of the MathMLNode.
        """
        attribute_str = " ".join(
            [f'{key}="{value}"' for key, value in node.attributes.items()]
        )
        if node.content is not None:
            return f"{node.tag} {attribute_str}: {node.content.strip()}"
        else:
            return f"{node.tag} {attribute_str}"

    zss_tree1 = zss.Node(get_label(mathml_tree1))
    zss_tree2 = zss.Node(get_label(mathml_tree2))

    def add_children(node: MathMLNode, zss_node: zss.Node) -> None:
        """
        Recursively adds children to a zss.Node.

        Args:
            node (MathMLNode): The MathMLNode to add children from.
            zss_node (zss.Node): The zss.Node to add children to.
        """
        for child in node.children:
            child_zss_node = zss.Node(get_label(child))
            zss_node.addkid(child_zss_node)
            add_children(child, child_zss_node)

    add_children(mathml_tree1, zss_tree1)
    add_children(mathml_tree2, zss_tree2)

    return zss.simple_distance(zss_tree1, zss_tree2)


def get_ted_loss(ted: float, sensitivity: float = 0.25) -> float:
    """
    Calculates the tree edit distance (TED) loss based on the given TED value and sensitivity.

    Args:
        ted (float): The tree edit distance value.
        sensitivity (float): The sensitivity parameter. Controls the range of the output.

    Returns:
        float: The calculated TED loss.

    """
    if ted == -1:
        return 1.0

    # Calculate the TED loss using the hyperbolic tangent function
    loss = math.tanh(ted * sensitivity)

    return loss


def get_batch_ted_loss(outputs: torch.Tensor, mml: torch.Tensor, vocab: Vocab) -> float:
    """
    Calculates the batch tree edit distance (TED) loss based on the given outputs, MathML tensor, and vocabulary.

    Args:
        outputs (torch.Tensor): Tensor containing the model outputs.
        mml (torch.Tensor): Tensor containing the MathML sequences.
        vocab (Vocab): Vocabulary object containing the mapping between tokens and indices.

    Returns:
        float: The calculated batch TED loss.

    """
    batch_size = outputs.size(0)
    output_dim = outputs.size(2)
    batch_ted_loss = 0

    for i in range(batch_size):
        # Convert output and MathML tensors to token sequences
        output_tokens = [
            vocab.itos[idx] for idx in torch.argmax(F.softmax(outputs[i], dim=1), dim=1)
        ]
        mml_tokens = [vocab.itos[idx] for idx in mml[i]]

        output_str = " ".join(output_tokens)
        mml_str = " ".join(mml_tokens)

        # Remove <EOS> and everything after it
        output_str = output_str.split(" <eos>", 1)[0]
        mml_str = mml_str.split(" <eos>", 1)[0]

        # Calculate tree edit distance and accumulate the TED loss
        distance = calculate_tree_edit_distance(output_str, mml_str)
        batch_ted_loss += get_ted_loss(distance)

    # Normalize the TED loss by the batch size
    batch_ted_loss = batch_ted_loss / batch_size
    return batch_ted_loss


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
    vocab=None,
    weight=0.5,
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
        batch_ted_loss = get_batch_ted_loss(outputs, mml[:, 1:], vocab)
        # avoiding <sos> token while Calculating loss
        mml = mml[:, 1:].contiguous().view(-1)
        if model_type == "opennmt":
            outputs = outputs[:, 1:, :].contiguous().view(-1, output_dim)
        elif model_type == "cnn_xfmer" or model_type == "resnet_xfmer":
            outputs = outputs.contiguous().view(-1, output_dim)

        loss = criterion(outputs, mml) + weight * batch_ted_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    net_loss = epoch_loss / len(train_dataloader)
    return net_loss
