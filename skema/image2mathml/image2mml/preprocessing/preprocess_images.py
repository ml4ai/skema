import torch, os, json, argparse
from torchvision import transforms
from PIL import Image
import multiprocessing
from multiprocessing import Pool, Lock, TimeoutError

# opening config file
parser = argparse.ArgumentParser()
parser.add_argument("--config", help="configuration file for paths and hyperparameters. for example\
                                        configs/ourmml_xfmer_config.json")
args = parser.parse_args()

with open(args.config, "r") as cfg:
    config = json.load(cfg)

def crop_image(image, size):
    return transforms.functional.crop(image, 0, 0, size[0], size[1])

def resize_image(image):
    return image.resize((int(image.size[0]*0.5), int(image.size[1]*0.5)))

def pad_image(IMAGE):
    # was 8 all
    right = 8
    left = 8
    top = 8
    bottom = 8
    width, height = IMAGE.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(IMAGE.mode, (new_width, new_height))
    result.paste(IMAGE,(left, top))
    return result

def preprocess_images(image):

    """
    RuntimeError: only Tensors of floating point dtype can require gradients
    Crop, padding, and downsample the image.
    We will modify the images according to the max size of the image/batch

    :params img_batch: batch of images
    :return: processed image tensor for enitre batch-[Batch, Channels, W, H]
    """

    IMAGE = Image.open(f"{config['data_path']}/{config['dataset_type']}/images/{image}").convert("L")

    # crop the image
    w, h = 500, 50
    IMAGE = crop_image(IMAGE, [h, w])

    # resize
    IMAGE = resize_image(IMAGE)

    # padding
    IMAGE = pad_image(IMAGE)

    # convert to tensor
    convert = transforms.ToTensor()
    IMAGE = convert(IMAGE)

    # saving the image
    torch.save(IMAGE, f"{config['data_path']}/{config['dataset_type']}/image_tensors/{image.split('.')[0]}.txt")

def main():

    data_path = f"{config['data_path']}/{config['dataset_type']}"
    images = os.listdir(f'{data_path}/images')

    # create an image_tensors folder
    if not os.path.exists(f'{data_path}/image_tensors'):
        os.mkdir(f'{data_path}/image_tensors')

    with Pool(multiprocessing.cpu_count()) as pool:
            result = pool.map(preprocess_images, images)


if __name__ == "__main__":
    main()
