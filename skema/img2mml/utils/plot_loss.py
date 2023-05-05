import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["arxiv", "im2mml", "arxiv_im2mml"],
    default="arxiv",
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
args = parser.parse_args()

plt.style.use("ggplot")
# def my_plot(epochs, loss):
#     plt.plot(epochs, loss)
#

dataset = args.mode
if args.with_fonts:
    dataset += "_with_fonts"
if args.with_boldface:
    dataset += "_boldface"

t_loss, v_loss = [], []
f = open(f"logs/{dataset}_loss_file.txt").readlines()

for i in f:
    if "Train Loss" in i:
        t_loss.append(float(i.split("|")[0].split(":")[1].strip()))
    elif "Val. Loss" in i:
        v_loss.append(float(i.split("|")[0].split(":")[1].strip()))

x = range(len(t_loss))
plt.figure(figsize=(12, 10))
plt.xticks(np.arange(0, len(t_loss), 10))
plt.yticks(np.arange(0, max(max(t_loss), max(v_loss)), 0.2))
plt.plot(x, t_loss, label="train")
plt.plot(x, v_loss, label="val")
plt.legend(loc="upper right")
plt.xlabel("num_epochs")
plt.ylabel("loss")
plt.savefig(f"logs/{dataset}_loss_fig.png")
