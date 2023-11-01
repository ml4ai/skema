import json
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import time
from shutil import copyfile as CP
from threading import Timer
from typing import List, Set, Tuple, Union

from tqdm import tqdm

# read config file and define paths
config_path = "sampling_dataset/sampling_config.json"
with open(config_path, "r") as cfg:
    config = json.load(cfg)

global final_paths, count, lock, total_eqns, verbose, chunk_size
global distribution_achieved, counter_dist_dict, dist_dict

verbose = config["verbose"]

# chink_size for multiprocessing
chunk_size = config["chunk_size"]

# src path
root = config["src_path"]

# setting seed
random.seed(config["seed"])

# fields we want papers from
fields = (
    set(config["fields"]) if "fields" in config and len(config["fields"]) > 0 else None
)

# create destination files and directory
if not os.path.exists("training_data"):
    os.mkdir("training_data")

data_path = "training_data/sample_data"

if not os.path.exists(data_path):
    os.mkdir(data_path)
else:
    print(
        "sample_data already exists. \
        Removing old sample_data and replacing \
        it with new one."
    )
    shutil.rmtree(data_path)
    os.mkdir(data_path)

os.mkdir(os.path.join(data_path, "images"))
mml_file = open(os.path.join(data_path, "original_mml.lst"), "w")
latex_file = open(os.path.join(data_path, "original_latex.lst"), "w")
paths_file = open(os.path.join(data_path, "paths.lst"), "w")

# distribution
dist_dict = dict()
counter_dist_dict = dict()
total_eqns = 0

# initialize the dist_dict
for i in tqdm(
    range(0, 350, 50),
    desc="Initializing distribution dict...",
):
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
lock = mp.Lock()


def read_latex(path: str) -> str:
    with open(path, "r") as f:
        latex_arr = f.readlines()
        latex = (
            " ".join([l.replace("\n", "") for l in latex_arr])
            if len(latex_arr) >= 1
            else ""
        )
    return latex


def eqn_image_exists(path: str) -> bool:
    yr, month, folder, type_of_eqn, eqn_num = path.split("_")
    img_path = os.path.join(
        root,
        f"{yr}/{month}/latex_images/{folder}/{type_of_eqn}_eqns/{eqn_num}.png",
    )
    return os.path.exists(img_path)


def eqn_mml_exists(path: str) -> bool:
    yr, month, folder, type_of_eqn, eqn_num = path.split("_")
    mml_path = os.path.join(
        root,
        f"{yr}/{month}/mathjax_mml/{folder}/{type_of_eqn}_mml/{eqn_num}.xml",
    )
    return os.path.exists(mml_path)


def get_paths(yr: str, yr_path: str, month: str) -> List[str]:
    """
    Collects equation paths for a given year and month.

    Args:
        yr (str): The year.
        yr_path (str): The path to the year directory.
        month (str): The month.

    Returns:
        List[str]: A list of equation paths in the format "yr_month_paper_type-of-eqn_eqn_num".
    """
    start_time = time.perf_counter()
    temp_files = []
    month_path = os.path.join(yr_path, month, "latex_equations")

    for paper in tqdm(
        os.listdir(month_path),
        desc=f"Collecting paths for {yr}/{month}",
        total=len(os.listdir(month_path)),
    ):
        for type_of_eqn in ("large", "small"):
            eqn_folder = os.path.join(month_path, paper, f"{type_of_eqn}_eqns")

            temp_files.extend(
                [
                    f"{yr}_{month}_{paper}_{type_of_eqn}_{eqn.split('.')[0]}"
                    for eqn in os.listdir(eqn_folder)
                ]
            )
    end_time = time.perf_counter()
    print(f"Finished collecting paths in {(end_time - start_time):.2f} seconds")
    return temp_files


def has_intersection(a: Set, b: Set) -> bool:
    return bool(a & b)


def filter_by_mml_existence(paths: List[str]) -> List[str]:
    """
    Filters a list of paths by whether or not the corresponding MathML file exists.

    Args:
        paths (List[str]): A list of paths in the format "yr_month_paper_type-of-eqn_eqn_num".

    Returns:
        List[str]: A list of paths in the format "yr_month_paper_type-of-eqn_eqn_num" for which the corresponding MathML file exists.
    """
    start_time = time.perf_counter()
    filtered_paths = []
    print("Filtering paths by mml existence. This may take a while...")
    with mp.Pool(config["num_cpus"]) as pool:
        results = pool.map(eqn_mml_exists, paths)
        for idx, result in tqdm(enumerate(results), total=len(results)):
            if result:
                filtered_paths.append(paths[idx])

    end_time = time.perf_counter()
    print(f"Trimmed {len(paths)} paths to {len(filtered_paths)} paths")
    print(f"Finished filtering paths in {end_time - start_time} seconds")
    return filtered_paths


def filter_by_image_existence(paths: List[str]) -> List[str]:
    """
    Filters a list of paths by whether or not the corresponding image file exists.

    Args:
        paths (List[str]): A list of paths in the format "yr_month_paper_type-of-eqn_eqn_num".

    Returns:
        List[str]: A list of paths in the format "yr_month_paper_type-of-eqn_eqn_num" for which the corresponding image file exists.
    """
    start_time = time.perf_counter()
    filtered_paths = []
    print("Filtering paths by image existence. This may take a while...")
    with mp.Pool(config["num_cpus"]) as pool:
        results = pool.map(eqn_image_exists, paths)
        for idx, result in tqdm(enumerate(results), total=len(results)):
            if result:
                filtered_paths.append(paths[idx])

    end_time = time.perf_counter()
    print(f"Trimmed {len(paths)} paths to {len(filtered_paths)} paths")
    print(f"Finished filtering paths in {end_time - start_time} seconds")
    return filtered_paths


def filter_paths_by_field(paths: List[str], fields: Set[str]) -> List[str]:
    """Filters a list of paths by whether or not the corresponding paper is in the specified fields.

    Args:
        paths (List[str]): A list of paths in the format "yr_month_paper_type-of-eqn_eqn_num".
        fields (Set[str]): A set of fields to filter by.

    Returns:
        List[str]: A list of paths in the format "yr_month_paper_type-of-eqn_eqn_num" for which the corresponding paper is in the specified fields.
    """
    start_time = time.perf_counter()
    print(f"Filtering paths by fields: {','.join(fields)}")
    arxvid_id_list = []
    for path in tqdm(paths, desc="Collecting arxiv ids from paths", total=len(paths)):
        arxiv_id = path.split("_")[2]
        arxvid_id_list.append(arxiv_id)

    script_dir = os.path.dirname(__file__)
    with open(
        os.path.join(script_dir, "paper_data", "arxiv_2015-2018_paper_categories.json"),
        "r",
    ) as f:
        category_dict = json.load(f)

    filtered_paths = []
    for path in tqdm(paths, desc="Filtering paths by fields", total=len(paths)):
        arxiv_id = path.split("_")[2]
        if arxiv_id in category_dict:
            paper_fields = {
                category.split(".")[0].lower() for category in category_dict[arxiv_id]
            }
            if has_intersection(paper_fields, fields):
                filtered_paths.append(path)
    end_time = time.perf_counter()
    print(f"Trimmed {len(paths)} paths to {len(filtered_paths)} paths")
    print(f"Finished filtering paths in {end_time - start_time} seconds")
    return filtered_paths


def divide_all_paths_into_chunks(all_paths: List[str]) -> List[List[str]]:
    global chunk_size
    for i in range(0, len(all_paths), chunk_size):
        yield all_paths[i : i + chunk_size]


def copy_image(img_src: str, img_dst: str) -> bool:
    try:
        CP(img_src, img_dst)
        return True
    except:
        return False


def normalize_latex(eqn_path: Tuple[int, str]) -> bool:
    """
    Runs latex normalization and simplification on a given equation. The equation is specified by the path.
    The normalized equation is stored in a temporary file in the temp_folder directory.

    For example, if the path is (1, "2001_01_2015.00001_1_large_eqn1"), then the equation is located at sampling_dataset/temp_folder/eqn_set_1.txt since the index is 1.

    Args:
        path (Tuple[int, str]): A tuple containing the index of the equation (in the batch) and the path to the equation in the format "yr_month_paper_type-of-eqn_eqn_num".

    Returns:
        bool: True if the equation was successfully normalized and simplified, False otherwise.
    """
    i, ap = eqn_path
    yr, month, folder, type_of_eqn, eqn_num = ap.split("_")
    latex_path = os.path.join(
        root,
        f"{yr}/{month}/latex_equations/{folder}/{type_of_eqn}_eqns/{eqn_num}.txt",
    )
    latex = read_latex(latex_path)

    temp_file_path = f"{os.getcwd()}/sampling_dataset/temp_folder/eqn_set_{i}.txt"
    with open(temp_file_path, "w") as f:
        f.write(latex)

    # run preprocess_latex.py
    cwd = os.getcwd()
    cmd = ["python", f"{cwd}/sampling_dataset/preprocess_latex.py", temp_file_path]
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    my_timer = Timer(10, kill, [process])

    # start the timer and run the process
    try:
        my_timer.start()

    except:
        if verbose:
            lock.acquire()
            print(f"Taking too long time. skipping {ap} equation...")
            lock.release()
        return False

    finally:
        my_timer.cancel()

    return True


def get_bin(eqn_path: Tuple[int, str]) -> Union[Tuple[str, str], None]:
    """
    Gets the distribution bin for a given equation. The equation is specified by the path. The bin is determined by the length of the normalized latex equation.
    The normalized equation file is located in the temp_folder directory using the index of the equation in the batch.

    Args:
        eqn_path (Tuple[int, str]): A tuple containing the index of the equation (in the batch) and the path to the equation in the format "yr_month_paper_type-of-eqn_eqn_num".

    Returns:
        Tuple[str, str]: A tuple containing the distribution bin and the path to the equation in the format "yr_month_paper_type-of-eqn_eqn_num".
    """
    try:
        i, ap = eqn_path

        latex = read_latex(
            f"{os.getcwd()}/sampling_dataset/temp_folder/eqn_set_{i}.txt"
        )
        latex_length = len(latex.split())

        # finding the bin
        temp_dict = {}
        for i in range(50, 400, 50):
            if latex_length / i < 1:
                temp_dict[i] = latex_length / i

        # get the bin
        if len(temp_dict) >= 1:
            max_bin_size = max(temp_dict, key=lambda k: temp_dict[k])
            tgt_bin = f"{max_bin_size-50}-{max_bin_size}"
        else:
            tgt_bin = "350+"

        return (tgt_bin, ap)

    except Exception as e:
        print(f"Error in {ap}:", e)
        return None


def kill(process):
    return process.kill()


def sample_paths_for_years(all_paths: List[str]) -> List[str]:
    """
    Collects equation paths for a given year.

    Args:
        all_paths (List[str]): A list of paths in the format "yr_month_paper_type-of-eqn_eqn_num".

    Returns:
        List[str]: A list of equation paths in the format "yr_month_paper_type-of-eqn_eqn_num".
    """
    years = [yr.strip() for yr in config["years"].split(",")]

    # Prepare batches for multiprocessing over both years and months
    batches = []
    for yr in years:
        yr_path = os.path.join(root, yr)
        cur_year_months = [f"{yr[2:]}{m:02}" for m in range(1, 13)]
        cur_year_batches = [(yr, yr_path, month) for month in cur_year_months]
        batches.extend(cur_year_batches)

    # Collect all paths for all the years in parallel
    with mp.Pool(config["num_cpus"]) as pool:
        results = [
            pool.apply_async(get_paths, args=(yr, yr_path, month))
            for yr, yr_path, month in batches
        ]

        for r in results:
            all_paths.extend(r.get())

    return all_paths


def sample_paths_for_months(all_paths: List[str]) -> List[str]:
    """
    Collects equation paths for a given month.

    Args:
        all_paths (List[str]): A list of paths in the format "yr_month_paper_type-of-eqn_eqn_num".

    Returns:
        List[str]: A list of equation paths in the format "yr_month_paper_type-of-eqn_eqn_num".
    """
    months = [month.strip() for month in config["months"].split(",")]
    corresponding_years = [f"20{month[:2]}" for month in months]

    # Preparing batches for multiprocessing over months
    batches = zip(corresponding_years, months)

    # Collect all paths for all the months in parallel
    with mp.Pool(config["num_cpus"]) as pool:
        results = [
            pool.apply_async(get_paths, args=(yr, os.path.join(root, yr), month))
            for yr, month in batches
        ]

        for r in results:
            all_paths.extend(r.get())

    return all_paths


def main():
    """
    The main function that runs the sampling process

    Sampling Steps:

        (1) All the equation paths will be collected according to the config.
        (2) They will be randomly shuffled twice to ensure proper shuffling.
        (3) They will sequentially be fetched and will be distributed according to the
            length as per the distribution. The length will be determined by the normalized latex.
        (4) The corresponding PNG and MathML will be copied to the destination folder.
    """
    global count, total_eqns, final_paths, fields
    global lock, counter_dist_dict, dist_dict, chunk_size

    all_paths = []

    ######## Step 1: get all LaTeX paths ########
    print("Collecting all the LaTeX paths...")

    if config["sample_entire_year"]:
        all_paths = sample_paths_for_years(all_paths)
    elif config["sample_from_months"]:
        all_paths = sample_paths_for_months(all_paths)

    all_paths = filter_paths_by_field(all_paths, fields) if fields else all_paths
    all_paths = filter_by_image_existence(all_paths)
    all_paths = filter_by_mml_existence(all_paths)

    ######## Step 2: shuffle it twice ########
    print("Shuffling all the paths to create randomness...")
    random.shuffle(all_paths)
    random.shuffle(all_paths)

    ######## Step 3: Normalize latex and get bins ########
    print("Normalizing dataset...")

    # Opening a temporary folder to store temp files
    # this folder will be deleted at the end of the run.
    # It is created to help expediting the process by avoiding
    # Lock functionality.
    temp_folder = f"{os.getcwd()}/sampling_dataset/temp_folder"
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)

    print("Diving all_paths into batches of 10K to work efficiently...")
    reject_count = 0
    batches = list(divide_all_paths_into_chunks(all_paths))
    for bidx, batch_paths in tqdm(
        enumerate(batches), desc="Batches", total=len(batches)
    ):
        if count <= total_eqns:
            print("Running batch: ", bidx)
            print("Current distribution status: ", counter_dist_dict)

            # Preparing dataset for multiprocessing
            all_files = [(i, ap) for i, ap in enumerate(batch_paths)]

            print("Normalizing latex...")
            with mp.Pool(config["num_cpus"]) as pool:
                pool.map(normalize_latex, all_files)

            # getting bins for each equation's latex token length
            print("Getting bins for each equation's latex token length...")
            with mp.Pool(config["num_cpus"]) as pool:
                results = [
                    pool.apply_async(get_bin, args=(i,)).get() for i in all_files
                ]

            print("Collecting final paths for this batch...")
            for r in results:
                if r is not None:
                    tgt_bin, ap = r
                    if counter_dist_dict[tgt_bin] <= dist_dict[tgt_bin]:
                        counter_dist_dict[tgt_bin] += 1
                        final_paths.append(ap)
                        count += 1
                else:
                    reject_count += 1

            clean_cmd = [
                "rm",
                "-rf",
                f"{os.getcwd()}/sampling_dataset/temp_folder/*",
            ]
            subprocess.run(clean_cmd)

        else:
            # Remove temp_folder
            shutil.rmtree(temp_folder)
            print("Rejected: ", reject_count)
            break

    ######## Step 4: writing the final dataset ########

    # random shuffle twice
    random.shuffle(final_paths)
    random.shuffle(final_paths)

    print("Writing the final dataset files and copying images...")

    c_idx, reject = 0, 0
    for _, fp in tqdm(
        enumerate(final_paths), desc="Writing files...", total=len(final_paths)
    ):
        try:
            yr, month, folder, type_of_eqn, eqn_num = fp.split("_")

            # copying image
            img_src = os.path.join(
                root,
                f"{yr}/{month}/latex_images/{folder}/{type_of_eqn}_eqns/{eqn_num}.png",
            )
            img_dst = os.path.join(os.path.join(data_path, "images"), f"{c_idx}.png")
            CP(img_src, img_dst)

            # Saving path
            paths_file.write(fp + "\n")

            # Saving mml
            mml_path = os.path.join(
                root,
                f"{yr}/{month}/mathjax_mml/{folder}/{type_of_eqn}_mml/{eqn_num}.xml",
            )
            with open(mml_path, "r") as f:
                mml = f.readline()

            mml = mml + "\n" if "\n" not in mml else mml
            mml_file.write(mml)

            # Saving latex
            latex_path = os.path.join(
                root,
                f"{yr}/{month}/latex_equations/{folder}/{type_of_eqn}_eqns/{eqn_num}.txt",
            )
            latex = read_latex(latex_path)
            latex = latex + "\n" if "\n" not in latex else latex
            latex_file.write(latex)

            c_idx += 1

        except Exception as e:
            reject += 1
            print("Rejected: with error: ", e)

    print("Final distribution: ", counter_dist_dict)
    print("Total equations: ", c_idx)
    print("Total rejected: ", reject)
