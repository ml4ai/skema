import argparse
import os
import shutil
import random

parser = argparse.ArgumentParser(
    description="Create the dataset for training and evaluation."
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
parser.add_argument("--seed", type=int, default=20, help="The random seed.")

args = parser.parse_args()


data_path = "training_data/sample_data/"
data_path = data_path + str(args.mode)
if args.with_fonts:
    data_path += "_with_fonts"
random.seed(args.seed)


def merge_sample_data(with_fonts=True):
    # Set up paths
    if with_fonts:
        arxiv_images_path = "training_data/arxiv_sample_data/images_fonts"
        im2mml_images_path = "training_data/im2mml-100K/images_fonts"
    else:
        arxiv_images_path = "training_data/arxiv_sample_data/images"
        im2mml_images_path = "training_data/im2mml-100K/images"
    arxiv_latex_path = "training_data/arxiv_sample_data/original_latex.lst"
    im2mml_latex_path = "training_data/im2mml-100K/latex.lst"
    arxiv_mml_path = "training_data/arxiv_sample_data/original_mml.lst"
    im2mml_mml_path = "training_data/im2mml-100K/original_mml.lst"
    sample_images_path = data_path + "/images"
    sample_latex_path = data_path + "/original_latex.lst"
    sample_mml_path = data_path + "/original_mml.lst"

    # Create sample_data directory and images subdirectory
    os.makedirs(sample_images_path, exist_ok=True)

    # Copy image files from arxiv_sample_data/images and im2mml-100K/images
    arxiv_image_files = os.listdir(arxiv_images_path)
    for filename in arxiv_image_files:
        shutil.copyfile(
            os.path.join(arxiv_images_path, filename),
            os.path.join(sample_images_path, filename),
        )
    im2mml_image_files = os.listdir(im2mml_images_path)
    im2mml_image_files = sorted(im2mml_image_files, key=lambda x: int(x.split(".")[0]))
    start_num = len(arxiv_image_files)
    for i, filename in enumerate(im2mml_image_files):
        shutil.copyfile(
            os.path.join(im2mml_images_path, filename),
            os.path.join(sample_images_path, f"{i + start_num}.png"),
        )

    # Merge original_latex.lst and latex.lst
    with open(arxiv_latex_path, "r") as arxiv_latex_file:
        with open(im2mml_latex_path, "r") as im2mml_latex_file:
            with open(sample_latex_path, "a") as sample_latex_file:
                sample_latex_file.writelines(arxiv_latex_file.readlines())
                sample_latex_file.write("\n")
                sample_latex_file.writelines(im2mml_latex_file.readlines())

    # Merge original_mml.lst and original_mml.lst
    with open(arxiv_mml_path, "r") as arxiv_mml_file:
        with open(im2mml_mml_path, "r") as im2mml_mml_file:
            with open(sample_mml_path, "a") as sample_mml_file:
                sample_mml_file.writelines(arxiv_mml_file.readlines())
                sample_mml_file.write("\n")
                sample_mml_file.writelines(im2mml_mml_file.readlines())

    # # Randomize image files
    # img_files = os.listdir(sample_images_path)
    # random.shuffle(img_files)
    # for i, img_file in enumerate(img_files):
    #     src_file = os.path.join(sample_images_path, img_file)
    #     dst_file = os.path.join(sample_images_path, f"{i}.png")
    #     os.rename(src_file, dst_file)
    #
    # # Update original_latex.lst and original_mml.lst
    # shutil.copyfile(
    #     os.path.join(data_path, "original_latex.lst"),
    #     os.path.join(data_path, "original_latex_copy.lst"),
    # )
    # with open(os.path.join(data_path, "original_latex_copy.lst"), "r") as in_file:
    #     latex_lines = in_file.readlines()
    # with open(os.path.join(data_path, "original_latex.lst"), "w") as f:
    #     f.write("")
    #
    # shutil.copyfile(
    #     os.path.join(data_path, "original_mml.lst"),
    #     os.path.join(data_path, "original_mml_copy.lst"),
    # )
    # with open(os.path.join(data_path, "original_mml_copy.lst"), "r") as in_file:
    #     mml_lines = in_file.readlines()
    # with open(os.path.join(data_path, "original_mml.lst"), "w") as f:
    #     f.write("")
    #
    # with open(os.path.join(data_path, "original_latex.lst"), "a") as out_file:
    #     with open(os.path.join(data_path, "original_mml.lst"), "a") as out_file2:
    #         for i, img_file in enumerate(img_files):
    #             out_file.write(latex_lines[int(img_file.split(".")[0])].strip() + "\n")
    #             out_file2.write(mml_lines[int(img_file.split(".")[0])].strip() + "\n")
    #
    # os.remove(os.path.join(data_path, "original_latex_copy.lst"))
    # os.remove(os.path.join(data_path, "original_mml_copy.lst"))


def main():
    #  If the sample dataset exists, remove it first.
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    else:
        shutil.rmtree(data_path)
        os.mkdir(data_path)

    if args.mode == "arxiv":
        if args.with_fonts:
            image_path = "training_data/arxiv_sample_data/images_fonts"
            shutil.copytree(image_path, data_path + "/images")
            # os.rename(data_path + "/images_fonts", data_path + "/images")
        else:
            image_path = "training_data/arxiv_sample_data/images"
            shutil.copytree(image_path, data_path + "/images")

        latex_path = "training_data/arxiv_sample_data/original_latex.lst"
        shutil.copy(latex_path, data_path)
        mml_path = "training_data/arxiv_sample_data/original_mml.lst"
        shutil.copy(mml_path, data_path)
        path_path = "training_data/arxiv_sample_data/paths.lst"
        shutil.copy(mml_path, path_path)

    elif args.mode == "im2mml":
        if args.with_fonts:
            image_path = "training_data/im2mml-100K/images_fonts"
            shutil.copytree(image_path, data_path + "/images")
            # os.rename(data_path + "/images_fonts", data_path + "/images")
        else:
            image_path = "training_data/im2mml-100K/images"
            shutil.copytree(image_path, data_path + "/images")
        latex_path = "training_data/im2mml-100K/latex.lst"
        shutil.copy(latex_path, data_path)
        mml_path = "training_data/im2mml-100K/original_mml.lst"
        shutil.copy(mml_path, data_path)
    else:
        merge_sample_data(with_fonts=args.with_fonts)


if __name__ == "__main__":
    main()
