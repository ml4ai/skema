import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")
# def my_plot(epochs, loss):
#     plt.plot(epochs, loss)
#

t_loss, v_loss = [], []
f = open("logs/loss_file.txt").readlines()

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
plt.savefig("logs/loss_fig.png")
