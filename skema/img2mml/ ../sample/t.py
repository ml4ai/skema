import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

transform = T.ToPILImage()

img = transform(torch.load("image_tensors/997.txt"))

img.save("image.png")
