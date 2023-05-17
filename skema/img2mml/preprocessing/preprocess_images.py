"""
inspired by  https://github.com/harvardnlp/im2markup/blob/master/scripts/utils/image_utils.py
"""

import torch, os, json, argparse, sys
import numpy as np
from torchvision import transforms
from PIL import Image
import multiprocessing
from multiprocessing import Pool, Lock, TimeoutError

# read config file
config_path = sys.argv[-1]
with open(config_path, "r") as cfg:
    config = json.load(cfg)


def crop_image(image, reject=False):
    # converting to np array
    image_arr = np.asarray(image, dtype=np.uint8)

    # find where the data lies
    indices = np.where(image_arr != 255)

    # see if image is not blank
    # if both arrays of indices are null: blank image
    # if either is not null: it is only line either horizontal or vertical
    # In any case, thse image will be treated as garbage and will be discarded.

    if (len(indices[0]) == 0) or (len(indices[1]) == 0):
        reject = True

    else:
        # get the boundaries
        x_min = np.min(indices[1])
        x_max = np.max(indices[1])
        y_min = np.min(indices[0])
        y_max = np.max(indices[0])

        # crop the image
        image = image.crop((x_min, y_min, x_max, y_max))

    return image, reject


def resize_image(image, resize_factor):
    image = image.resize(
        (
            int(image.size[0] * resize_factor),
            int(image.size[1] * resize_factor),
        ),
        Image.Resampling.LANCZOS,
    )
    return image


def pad_image(image):
    pad = config["padding"]
    width = config["preprocessed_image_width"]
    hgt = config["preprocessed_image_height"]
    new_image = Image.new("RGB", (width, hgt), (255, 255, 255))
    new_image.paste(image, (pad, pad))
    return new_image


def bucket(image):
    """
    selecting the bucket based on the width, and hgt
    of the image. This will provide us the appropriate
    resizing factor.
    """
    # [width, hgt, resize_factor]
    buckets = [
        [820, 86, 0.6],
        [615, 65, 0.8],
        [492, 52, 1],
        [410, 43, 1.2],
        [350, 37, 1.4],
    ]
    # current width, hgt
    crop_width, crop_hgt = image.size[0], image.size[1]

    # find correct bucket
    resize_factor = config["resizing_factor"]
    for b in buckets:
        w, h, r = b
        if crop_width <= w and crop_hgt <= h:
            resize_factor = r

    return resize_factor


def downsampling(image):
    """
    if the image is too large and we won't be
    able to do bucketing, in that case, we need
    to dowmsample the image first and then
    will proceed with the preprocessing.
    It will be helpful if some random images
    will be send as input.
    """
    w, h = image.size
    max_h = config["max_input_hgt"]
    # we have come up with this number
    # from the buckets dimensions
    if h >= max_h:
        # need to calculate the ratio
        resize_factor = max_h / h

    image = image.resize(
        (
            int(image.size[0] * resize_factor),
            int(image.size[1] * resize_factor),
        ),
        Image.Resampling.LANCZOS,
    )
    return image


def preprocess_images(image):
    """
    RuntimeError: only Tensors of floating point dtype can require gradients
    Crop, padding, and downsample the image.
    We will modify the images according to the max size of the image/batch

    :params img_batch: batch of images
    :return: processed image tensor for enitre batch-[Batch, Channels, W, H]
    """

    IMAGE = Image.open(
        f"{config['data_path']}/{config['dataset_type']}/images/{image}"
    ).convert("L")

    # checking the size of the image
    w, h = IMAGE.size
    if h >= config["max_input_hgt"]:
        IMAGE = downsampling(IMAGE)

    # crop the image
    IMAGE, reject = crop_image(IMAGE)

    if not reject:
        # bucketing
        resize_factor = bucket(IMAGE)

        # resize
        IMAGE = resize_image(IMAGE, resize_factor)

        # padding
        IMAGE = pad_image(IMAGE)

        # convert to tensor
        convert = transforms.ToTensor()
        IMAGE = convert(IMAGE)

        # saving the image
        torch.save(
            IMAGE,
            f"{config['data_path']}/{config['dataset_type']}/image_tensors/{image.split('.')[0]}.txt",
        )
        return None

    else:
        return image


def main():
    data_path = f"{config['data_path']}/{config['dataset_type']}"
    images = os.listdir(f"{data_path}/images")

    # create an image_tensors folder
    if not os.path.exists(f"{data_path}/image_tensors"):
        os.mkdir(f"{data_path}/image_tensors")

    with Pool(config["num_cpus"]) as pool:
        result = pool.map(preprocess_images, images)

    blank_images = [i for i in result if i != None]

    with open("logs/blank_images.lst", "w") as out:
        out.write("\n".join(str(item) for item in blank_images))

    # renaming the final image_tensors to make sequential
    tnsrs = sorted(
        [
            int(i.split(".")[0])
            for i in os.listdir(f"{data_path}/image_tensors")
        ]
    )
    os.chdir(f"{data_path}/image_tensors")
    for t in range(len(tnsrs)):
        os.rename(f"{tnsrs[t]}.txt", f"{t}.txt")


if __name__ == "__main__":
    main()
