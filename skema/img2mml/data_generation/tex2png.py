# Render PNGs from Tex files using pdflatex and pdf2image

import os, subprocess, random
import logging
import json
import multiprocessing
import time
from datetime import datetime
from multiprocessing import Pool, Lock, TimeoutError
from threading import Timer
from pdf2image import convert_from_path

# Printing starting time
print(" ")
start_time = datetime.now()
print("Starting at:  ", start_time)

# Defining global lock
lock = Lock()

# read config file and define paths
config_path = "data_generation_config.json"
with open(config_path, "r") as cfg:
    config = json.load(cfg)

src_path = config["source_directory"]
destination = config["destination_directory"]
directories = config["months"].split(",")
years = config["years"].split(",")
verbose = config["verbose"]

def main(year):

    src_path = config["source_directory"]
    destination = config["destination_directory"]
    directories = config["months"].split(",")
    verbose = config["verbose"]

    # Setting up Logger - To get log files
    logger = logging.getLogger()
    log_format = "%(levelname)s:%(message)s"

    logging.basicConfig(
        filename=f"tex2png{year}.log",
        level=logging.DEBUG,
        format=log_format,
        filemode="w",
    )



    for month_dir in directories:
        month_dir = str(month_dir)
        path = os.path.join(destination, f"{year}/{month_dir}")
        latex_images = os.path.join(path, "latex_images")
        if not os.path.exists(latex_images):
            subprocess.call(["mkdir", latex_images])

        pool_path(path)


def pool_path(path):

    global lock

    # Folder path to TeX files
    tex_folder_path = os.path.join(path, "tex_files")

    for folder in os.listdir(tex_folder_path):

        # make results PNG directories
        pdf_dst_root = os.path.join(path, f"latex_images/{folder}")
        pdf_large = os.path.join(pdf_dst_root, "Large_eqns")
        pdf_small = os.path.join(pdf_dst_root, "Small_eqns")
        for f in [pdf_dst_root, pdf_large, pdf_small]:
            if not os.path.exists(f):
                subprocess.call(["mkdir", f])

        # Paths to Large and Small TeX files
        large_tex_files = os.path.join(tex_folder_path, f"{folder}/Large_eqns")
        small_tex_files = os.path.join(tex_folder_path, f"{folder}/Small_eqns")

        for type_of_folder in [large_tex_files, small_tex_files]:
            pdf_dst = (
                pdf_large if type_of_folder == large_tex_files else pdf_small
            )

            # array to store pairs of [type_of_folder, file in type_of_folder] Will be used as arguments in pool.map
            temp = []
            for texfile in os.listdir(type_of_folder):
                temp.append([folder, type_of_folder, texfile, pdf_dst])

            with Pool(multiprocessing.cpu_count()-200) as pool:
                result = pool.map(run_pdflatex, temp)


# This function will run pdflatex
def run_pdflatex(run_pdflatex_list):

    global lock

    (folder, type_of_folder, texfile, pdf_dst) = run_pdflatex_list

    os.chdir(pdf_dst)
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        os.path.join(type_of_folder, texfile),
    ]

    output = subprocess.Popen(
        command, stderr=subprocess.PIPE, stdout=subprocess.PIPE
    )
    my_timer = Timer(5, kill, [output])

    try:
        my_timer.start()
        stdout, stderr = output.communicate()

        # Calling pdf2png
        pdf2png(
            folder,
            f'{texfile.split(".")[0]}.pdf',
            texfile.split(".")[0],
            pdf_dst,
            type_of_folder,
        )

    finally:
        my_timer.cancel()


# Function to convert PDFs to PNGs
def pdf2png(folder, pdf_file, png_name, png_dst, type_of_folder):

    global lock

    os.chdir(png_dst)

    try:

        command_args = [
            "convert",
            "-background",
            "white",
            "-alpha",
            "remove",
            "off",
            "-density",
            "200",
            "-quality",
            "100",
            pdf_file,
            f"{png_dst}/{png_name}.png",
        ]

        subprocess.Popen(
            command_args, stderr=subprocess.PIPE, stdout=subprocess.PIPE
        )

        # Removing log and aux file if exist

        os.remove(f'{pdf_file.split(".")[0]}.log')

        try:
            os.remove(f'{pdf_file.split(".")[0]}.aux')
        except:

            if verbose:
                lock.acquire()
                print(f'{pdf_file.split(".")[0]}.aux doesn\'t exists.')
                lock.release()

            lock.acquire()
            logger.warning(
                f'{folder}:{type_of_folder}:{pdf_file.split(".")[0]}.aux doesn\'t exists.'
            )
            lock.release()

    except:

        if verbose:
            lock.acquire()
            print(
                f"This {folder}:{png_dst}:{pdf_file} file couldn't convert to png."
            )
            lock.release()

        lock.acquire()
        logger.warning(
            f"{folder}:{png_dst}:{pdf_file} file couldn't convert to png."
        )
        lock.release()


# Function to kill process if TimeoutError occurs
kill = lambda process: process.kill()


if __name__ == "__main__":

    for year in years:
        main(str(year))

    # Printing stoping time
    print(" ")
    stop_time = datetime.now()
    print("Stoping at:  ", stop_time)
    print(" ")
    print("Rendering PNGs -- completed.")
