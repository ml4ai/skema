import os, json, random
import multiprocessing as mp
import shutil
from shutil import copyfile as CP
from preprocessing.preprocess_mml import simplification

# read config file and define paths
config_path = "sampling_dataset/sampling_config.json"
with open(config_path, "r") as cfg:
    config = json.load(cfg)

# src path
root = config["src_path"]

# setting seed
random.seed(config["seed"])

# create destination files and directory
data_path = "training_data/sample_data"
images_path = os.path.join(data_path, "images")

if not os.path.exists(data_path):
    os.mkdir(data_path)

if not os.path.exists(images_path):
    os.mkdir(images_path)
else:
    print(
        "sample_data already exists. Removing old sample_data and replacing it with new one."
    )
    shutil.rmtree(images_path)
    os.mkdir(images_path)

mml_file = open(os.path.join(data_path, "original_mml.lst"), "w")
latex_file = open(os.path.join(data_path, "original_latex.lst"), "w")
paths_file = open(os.path.join(data_path, "paths.lst"), "w")

# distribution
dist_dict = dict()
counter_dist_dict = dict()
total_eqns = 0

# initialize the dist_dict
for i in range(0, 350, 50):
    begin = str(i)
    end = str(i + 50)
    key = f"{begin}-{end}"
    dist_dict[key] = list()
    # total_eqns += config[f"{begin}-{end}"]
    counter_dist_dict[key] = config[f"{begin}-{end}"]
dist_dict["350+"] = list()
# total_eqns += config["350+"]
counter_dist_dict["350+"] = config["350+"]


def get_paths(months):

    mp_temp = list()

    for month in months:
        yr = f"20{month[0:2]}"
        yr_path = os.path.join(root, yr)
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
                    mp_temp.append([
                                mml_eqn_path,
                                os.path.join(folder_path, f"{tyf}/{eqn}")
                                ])

    print("pooling")
    p = mp.Pool(config["num_cpus"])
    for result in p.imap_unordered(get_lengths, mp_temp):
        print(result)
        if result < 0:
            print('terminating')
            p.terminate()
            break
    # with mp.Pool(config["num_cpus"]) as pool:
    #     result = pool.map(get_lengths, mp_temp)
    #     if result == 0:
    #         pool.terminate()
            # break


def get_lengths(args):

    [path, eqn_path] = args
    # getting the length
    mml = open(eqn_path).readlines()[0]
    simp_mml = simplification(mml)
    length_mml = len(simp_mml.split())

    # finding the bin
    temp_dict = {}
    for i in range(50, 400, 50):
        if length_mml / i < 1:
            temp_dict[i] = length_mml / i

    # get the bin
    if len(temp_dict) >= 1:
        max_bin_size = max(temp_dict, key=lambda k: temp_dict[k])
        tgt_bin = f"{max_bin_size-50}-{max_bin_size}"
    else:
        tgt_bin = "350+"

    dist_dict[tgt_bin].append(path)
    counter_dist_dict[tgt_bin] -= 1

    return sum(counter_dist_dict.values())

def prepare_dataset():

    print("preparing dataset...")

    for i in range(0, 350, 50):
        begin = str(i)
        end = str(i + 50)
        key = f"{begin}-{end}"
        N = config[key]

        paths = random.shuffle(dist_dict[key])

        for n in N:
            yr, month, folder, type_of_eqn, eqn_num  = paths[n].split()

            # writing MathML
            mml_path = os.path.join(
                root,
                f"{yr}/{month}/mathjax_mml/{folder}/{type_of_eqn}_mml/{eqn_num}.xml",
            )

            mml = open(mml_path).readlines()[0]

            if "\n" not in mml:
                mml = mml + "\n"
            mml_file.write(mml)

            # wrting path
            paths_file.write(paths[n] + "\n")

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

            # copying image
            img_src = os.path.join(
                root,
                f"{yr}/{month}/latex_images/{folder}/{type_of_eqn}_eqns/{eqn_num}.png",
            )
            img_dst = os.path.join(images_path, f"{count}.png")
            CP(img_src, img_dst)


def main():

    ######## step 1: get all MathML paths ########
    print("collecting all the MathML paths...")

    if config["sample_entire_year"]:
        years = config[years].split(",")

        for yr in years:
            yr = yr.strip()
            yr_path = os.path.join(root, yr)
            months = list()
            for m in range(1, 13, 1):
                months.append(yr[0:2] + f"{m:02}")

    elif config["sample_from_months"]:
        months = config["months"].split(",")

    get_paths(months)
    prepare_dataset()
