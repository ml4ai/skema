"""
inspired by  https://github.com/harvardnlp/im2markup/blob/master/scripts/utils/image_utils.py
"""

import torch, os, json, argparse
import numpy as np
from PIL.Image import Image
from torchvision import transforms
from PIL import Image, ImageEnhance
import multiprocessing
from multiprocessing import Pool, Lock, TimeoutError
import sys
import random

parser = argparse.ArgumentParser(
    description="Preprocess the images in the dataset for training and evaluation."
)
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
    "--enhance_images",
    action="store_true",
    default=False,
    help="Whether enhancing images",
)
parser.add_argument(
    "--config",
    type=str,
    default="configs/xfmer_mml_config.json",
    help="The configuration file.",
)

args = parser.parse_args()

config = None


def get_config(config_path):
    with open(config_path, "r") as cfg:
        config = json.load(cfg)

    return config


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
        [703, 74, 0.7],
        [615, 65, 0.8],
        [547, 58, 0.9],
        [492, 52, 1],
        [447, 47, 1.1],
        [410, 43, 1.2],
        [379, 40, 1.3],
        [350, 37, 1.4],
        [328, 35, 1.5],
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


def enhance_image(image: Image) -> Image:
    """
    Apply image enhancement techniques to the input image.

    Args:
        image (Image): The input image.

    Returns:
        Image: The enhanced image.
    """
    # Brightness enhancement
    brightness_factor = random.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)

    # Contrast enhancement
    contrast_factor = random.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    # Sharpness enhancement
    sharpness_factor = random.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness_factor)

    # # Image rotation
    # rotation_angle = random.randint(-10, 10)
    # image = image.rotate(rotation_angle)
    #
    # # Adding noise
    # noise_factor = random.uniform(0.01, 0.05)
    # width, height = image.size
    # noise = Image.frombytes('L', (width, height), bytes([random.randint(0, 255) for _ in range(width * height)]))
    # image = Image.blend(image, noise, noise_factor)

    return image


def preprocess_images(image):
    """
    RuntimeError: only Tensors of floating point dtype can require gradients
    Crop, padding, and downsample the image.
    We will modify the images according to the max size of the image/batch

    :params img_batch: batch of images
    :return: processed image tensor for enitre batch-[Batch, Channels, W, H]
    """
    data_path = f"training_data/sample_data/{args.mode}"
    if args.with_fonts:
        data_path += "_with_fonts"

    IMAGE = Image.open(f"{data_path}/images/{image}").convert("L")

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

        # if enhancing images
        if args.enhance_images:
            IMAGE = enhance_image(IMAGE)

        # convert to tensor
        convert = transforms.ToTensor()
        IMAGE = convert(IMAGE)

        # saving the image
        torch.save(
            IMAGE,
            f"{data_path}/image_tensors/{image.split('.')[0]}.txt",
        )
        return None

    else:
        return image


def main():
    global config
    config = get_config(args.config)
    random.seed(int(config["seed"]))

    data_path = f"training_data/sample_data/{args.mode}"
    if args.with_fonts:
        data_path += "_with_fonts"

    images = os.listdir(f"{data_path}/images")

    # create an image_tensors folder
    if not os.path.exists(f"{data_path}/image_tensors"):
        os.mkdir(f"{data_path}/image_tensors")

    with Pool(config["num_cpus"]) as pool:
        result = pool.map(preprocess_images, images)

    blank_images = [i for i in result if i != None]

    mode_name = args.mode
    if args.with_fonts:
        mode_name += "_with_fonts"

    with open(f"logs/{mode_name}_blank_images.lst", "w") as out:
        out.write("\n".join(str(item) for item in blank_images))

    # renaming the final image_tensors to make sequential
    tnsrs = sorted(
        [int(i.split(".")[0]) for i in os.listdir(f"{data_path}/image_tensors")]
    )
    os.chdir(f"{data_path}/image_tensors")
    for t in range(len(tnsrs)):
        os.rename(f"{tnsrs[t]}.txt", f"{t}.txt")


if __name__ == "__main__":
    main()
