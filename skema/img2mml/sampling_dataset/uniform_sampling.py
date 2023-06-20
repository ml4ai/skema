import os, json, random
import shutil
from shutil import copyfile as CP

# read config file and define paths
config_path = "sampling_dataset/sampling_config.json"
with open(config_path, "r") as cfg:
    config = json.load(cfg)

verbose = config["verbose"]

# src path
root = config["src_path"]

# setting seed
random.seed(config["seed"])

total_eqns = 100000  # Collect 100000 equations
final_paths = list()
count = 0


def get_paths(yr, yr_path, month):
    temp_files = list()

    month_path = os.path.join(yr_path, month)
    mml_path = os.path.join(month_path, "mathjax_mml")
    folders = os.listdir(mml_path)
    for folder in folders:
        folder_path = os.path.join(mml_path, folder)
        for tyf in os.listdir(folder_path):
            type_of_eqn = "large" if tyf == "large_mml" else "small"
            for eqn in os.listdir(os.path.join(folder_path, tyf)):
                eqn_num = eqn.split(".")[0]
                mml_eqn_path = f"{yr}_{month}_{folder}_{type_of_eqn}_{eqn_num}"

                temp_files.append(mml_eqn_path)

    return temp_files


def copy_image(img_src, img_dst):
    try:
        CP(img_src, img_dst)
        return True
    except:
        return False


# Load the boldface list
def process_boldface_list(filepath):
    def convert_filename(filepath):
        split_path = filepath.split("/")
        year = split_path[5]
        month = split_path[6]
        paper = split_path[8]
        size = split_path[9].split("_")[0]
        eqn_num = split_path[10].split(".")[0]
        new_filename = f"{year}_{month}_{paper}_{size}_{eqn_num}"

        return new_filename

    # Read file
    with open(filepath) as f:
        lines = f.readlines()

    # Process each line
    processed_names = []
    for line in lines:
        line = line.strip()
        if line != "":
            new_name = convert_filename(line)
            processed_names.append(new_name)

    return processed_names


def create_dataset():
    print("Generating the arxiv dataset")
    # create destination files and directory
    data_path = "training_data/arxiv_sample_data"
    images_path = os.path.join(data_path, "images")
    images_path_fonts = os.path.join(data_path, "images_fonts")

    if not os.path.exists(data_path):
        os.mkdir(data_path)
        os.mkdir(images_path)
        os.mkdir(images_path_fonts)
    else:
        shutil.rmtree(data_path)
        os.mkdir(data_path)
        os.mkdir(images_path)
        os.mkdir(images_path_fonts)
        print(
            "arxiv_sample_data already exists. Removing old sample_data and replacing it with new one."
        )

    mml_file = open(os.path.join(data_path, "original_mml.lst"), "w")
    latex_file = open(os.path.join(data_path, "original_latex.lst"), "w")
    paths_file = open(os.path.join(data_path, "paths.lst"), "w")

    boldface_list = process_boldface_list("data_generation/boldface_list.txt")

    """
    Sampling Steps:

    (1) All the MathML eqn paths will be collected in 'all_paths'.
    (2) They will be randomly shuffled twice to ensure proper shuffling.
    (3) They will sequentially be fetched. In the arvix dateset, we will ensure 50K equations with boldface.
    (4) Corresponding Latex and PNG will fetched and stored.

    """

    all_paths = list()

    ######## step 1: get all MathML paths ########
    print("collecting all the MathML paths...")

    if config["sample_entire_year"]:
        years = config["years"].split(",")
        for yr in years:
            yr = yr.strip()
            yr_path = os.path.join(root, yr)

            for m in range(1, 13, 1):
                month = yr[2:] + f"{m:02}"  # yr=2018, month=1801,..
                temp_paths = get_paths(yr, yr_path, month)
                for p in temp_paths:
                    all_paths.append(p)

    elif config["sample_from_months"]:
        months = config["months"].split(",")
        for month in months:
            month = month.strip()
            yr = f"20{month[0:2]}"
            yr_path = os.path.join(root, yr)
            temp_paths = get_paths(yr, yr_path, month)
            for p in temp_paths:
                all_paths.append(p)

    ######## step 2: shuffle it twice ########
    print("shuffling all the paths to create randomness...")
    random.shuffle(all_paths)
    random.shuffle(all_paths)
    random.shuffle(boldface_list)
    random.shuffle(boldface_list)

    ######## step 3: grab the corresponding PNG and latex  #####
    print("preparing dataset...")

    # Add 80000 equations with boldface to the dataset
    if len(boldface_list) > 80000:
        final_paths = boldface_list[:80000]
        count = 80000
    else:
        final_paths = boldface_list
        count = len(boldface_list)

    boldface_list_set = set(boldface_list)
    for path in all_paths:
        if count <= total_eqns:
            if count % 1000 == 0:
                print("{} equations collected.".format(count))

            # collect equations without boldface
            if path not in boldface_list_set:
                final_paths.append(path)
                count += 1
        else:
            break

    ######## step 4: writing the final dataset ########

    # random shuffle twice
    random.shuffle(final_paths)
    random.shuffle(final_paths)

    print("writing the final dataset files and copying images...")

    reject = 0
    c_idx = 0
    for fpidx, fp in enumerate(final_paths):
        try:
            yr, month, folder, type_of_eqn, eqn_num = fp.split("_")

            # copying image
            img_src = os.path.join(
                root,
                f"{yr}/{month}/latex_images/{folder}/{type_of_eqn}_eqns/{eqn_num}.png",
            )
            img_dst = os.path.join(images_path, f"{c_idx}.png")
            CP(img_src, img_dst)

            # copying image with diverse fonts
            img_src_fonts = os.path.join(
                root,
                f"{yr}/{month}/latex_images_fonts/{folder}/{type_of_eqn}_eqns/{eqn_num}.png",
            )
            img_dst_fonts = os.path.join(images_path_fonts, f"{c_idx}.png")
            CP(img_src_fonts, img_dst_fonts)

            # wrting path
            paths_file.write(fp + "\n")

            # writing MathML
            mml_path = os.path.join(
                root,
                f"{yr}/{month}/mathjax_mml/{folder}/{type_of_eqn}_mml/{eqn_num}.xml",
            )

            mml = open(mml_path).readlines()[0]

            if "\n" not in mml:
                mml = mml + "\n"
            mml_file.write(mml)

            # writing latex
            latex_path = os.path.join(
                root,
                f"{yr}/{month}/latex_equations/{folder}/{type_of_eqn}_eqns/{eqn_num}.txt",
            )
            latex_arr = open(latex_path).readlines()
            if len(latex_arr) > 1:
                latex = " "
                for l in latex_arr:
                    latex = latex + l.replace("\n", "")
            else:
                latex = latex_arr[0]

            if "\n" not in latex:
                latex = latex + "\n"
            latex_file.write(latex)

            c_idx += 1

        except:
            reject += 1
            pass

    print("total equations: ", c_idx)