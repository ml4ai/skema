import os, json, random
import shutil
import threading
import subprocess
from threading import Timer
from shutil import copyfile as CP
from preprocessing.preprocess_mml import simplification
import multiprocessing as mp

# read config file and define paths
config_path = "sampling_dataset/sampling_config.json"
with open(config_path, "r") as cfg:
    config = json.load(cfg)

global final_paths, count, lock, total_eqns, distribution_achieved, counter_dist_dict, dist_dict

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
    dist_dict[key] = config[f"{begin}-{end}"]
    total_eqns += config[f"{begin}-{end}"]
    counter_dist_dict[key] = 0
dist_dict["350+"] = config["350+"]
total_eqns += config["350+"]
counter_dist_dict["350+"] = 0

final_paths = list()
count = 0
distribution_achieved = False
lock = mp.Lock()

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

def prepare_dataset(args):

    global count, total_eqns, distribution_achieved, final_paths, lock, counter_dist_dict, dist_dict

    i, ap = args

    # print(i, ap)

    if (count <= total_eqns) and (not distribution_achieved):

        yr, month, folder, type_of_eqn, eqn_num = ap.split("_")
        mml_path = os.path.join(
            root,
            f"{yr}/{month}/mathjax_mml/{folder}/{type_of_eqn}_mml/{eqn_num}.xml",
        )

        mml = open(mml_path).readlines()[0]
        open(f"{os.getcwd()}/sampling_dataset/temp_folder/smr_{i}.txt", "w").write(mml)

        cwd = os.getcwd()
        cmd = ["python", f"{cwd}/sampling_dataset/simp.py", "--idx", str(i)]
        output = subprocess.Popen(
            cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE
        )
        my_timer = Timer(5, kill, [output])

        # try:
            # my_timer.start()
        stdout, stderr = output.communicate()
        simp_mml = open(f"{os.getcwd()}/sampling_dataset/temp_folder/sm_{i}.txt").readlines()[0]
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

        if counter_dist_dict[tgt_bin] <= dist_dict[tgt_bin]:
            counter_dist_dict[tgt_bin] += 1
            final_paths.append(ap)
            count+=1

        # except:
        #     lock.acquire()
        #     print("current status: ", counter_dist_dict)
        #     print(f"taking too long time. skipping {ap} equation...")
        #     lock.release()
        #
        # finally:
        #     my_timer.cancel()

    else:
        distribution_achieved = True


# Function to kill process if TimeoutError occurs
kill = lambda process: process.kill()


def main():

    """
    Sampling Steps:

    (1) All the MathML eqn paths will be collected in 'all_paths'.
    (2) They will be randomly shuffled twice to ensure proper shuffling.
    (3) They will sequentially be fetched and will be distributed according to the
        length as per the distribution. The length of the preprocessed simplified
        MathML used will be used to final distribution bin.
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

    ######## step 3: simplify MML and and find length  #####
    ######## and grab the corresponding PNG and latex ############
    print("preparing dataset...")

    # opening a temporary folder to store temp files
    # this folder will be deleted at the end of the run.
    # It is created to help expediting the process by avoiding
    # Lock functionality.
    temp_folder = f"{os.getcwd()}/sampling_dataset/temp_folder"
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)

    all_files = [[i, ap] for i, ap in enumerate(all_paths)]
    with mp.Pool(config["num_cpus"]) as pool:
        result = pool.map(prepare_dataset, all_files)

    # remove temp_folder
    shutil.rmtree(temp_folder)

    ######## step 4: writing the final dataset ########

    global final_paths

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

            c_idx+=1

        except:
            reject +=1
            pass
