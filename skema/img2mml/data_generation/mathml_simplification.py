# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 21:24:26 2020

@author: gauravs
"""

import re
import subprocess, os
import multiprocessing
import logging
from datetime import datetime
from multiprocessing import Pool, Lock, TimeoutError

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

    # Setting up Logger - To get log files
    log_format = "%(message)s"
    logfile_dst = os.path.join(destination, f"{year}/Logs")
    begin_month, end_month = directories[0], directories[-1]
    logging.basicConfig(
        filename=os.path.join(
            logfile_dst, f"{begin_month}-{end_month}_Unicode_MML.log"
        ),
        level=logging.DEBUG,
        format=log_format,
        filemode="w",
    )

    unicode_logger = logging.getLogger()

    for month_dir in directories:

        month_dir = str(month_dir)

        print("Directory running:  ", month_dir)

        month_dir_path = os.path.join(destination, f"{year}/{month_dir}")
        mathjax_mml_path = os.path.join(month_dir_path, "Mathjax_mml")

        # Making new directory for Simplified MML
        simp_mml_path = os.path.join(month_dir_path, "Simplified_mml")
        if not os.path.exists(simp_mml_path):
            subprocess.call(["mkdir", simp_mml_path])

        # Creating array fro pooling
        temp = []

        for folder in os.listdir(mathjax_mml_path):
            temp.append(
                [os.path.join(mathjax_mml_path, folder), folder, simp_mml_path]
            )

        with Pool(multiprocessing.cpu_count() - 30) as pool:
            result = pool.map(simplification, temp)


def simplification(pooling_list):

    global lock

    # Unpacking Pooling list of arguments
    (folder_path, folder, simp_mml_path) = pooling_list

    # Making directory named <folder> in simp_mml_path
    simp_mml_folder_path = os.path.join(simp_mml_path, folder)
    if not os.path.exists(simp_mml_folder_path):
        subprocess.call(["mkdir", simp_mml_folder_path])

    for type_of_folder in ["Large_MML", "Small_MML"]:

        type_of_folder_path = os.path.join(folder_path, type_of_folder)

        # Making directories in Simplified MML
        tyf_path = os.path.join(simp_mml_folder_path, type_of_folder)
        if not os.path.exists(tyf_path):
            subprocess.call(["mkdir", tyf_path])

        for file_path in os.listdir(type_of_folder_path):

            mml_org_line_list = open(
                os.path.join(type_of_folder_path, file_path), "r"
            ).readlines()

            if len(mml_org_line_list) > 0:

                mml_org = mml_org_line_list[0]

                if verbose:
                    lock.acquire()
                    # Printing original MML
                    print("Curently running folder:  ", folder)
                    print(" ")
                    print(
                        "=============== Printing original MML ================"
                    )
                    print("Original: \n")
                    print(mml_org)
                    lock.release()

                # Removing multiple backslashes
                i = mml_org.find("\\\\")
                mml_org = mml_org.encode().decode("unicode_escape")

                while i > 0:
                    mml_org = mml_org.replace("\\\\", "\\")
                    i = mml_org.find("\\\\")

                # Removing initial information about URL, display, and equation itself
                begin = mml_org.find("<math") + len("<math")
                end = mml_org.find(">")
                mml_org = mml_org.replace(mml_org[begin:end], "")

                # Checking and logging unicodes along with their asciii-code
                unicode_logger(mml_org, os.path.join(type_of_folder_path, file_path))

                # ATTRIBUTES

                ## Attributes commonly used in MathML codes to represent equations
                elements = [
                    "mrow",
                    "mi",
                    "mn",
                    "mo",
                    "ms",
                    "mtext",
                    "math",
                    "mtable",
                    "mspace",
                    "maction",
                    "menclose",
                    "merror",
                    "mfenced",
                    "mfrac",
                    "mglyph",
                    "mlabeledtr",
                    "mmultiscripts",
                    "mover",
                    "mroot",
                    "mpadded",
                    "mphantom",
                    "msqrt",
                    "mstyle",
                    "msub",
                    "msubsup",
                    "msup",
                    "mtd",
                    "mtr",
                    "munder",
                    "munderover",
                    "semantics",
                ]

                ## Attributes that can be removed
                attr_tobe_removed = [
                    "class",
                    "id",
                    "style",
                    "href",
                    "mathbackground",
                    "mathcolor",
                ]

                ## Attributes that need to be checked before removing, if mentioned in code with their default value,
                ## will be removed else will keep it. This dictionary contains all the attributes with thier default values.
                attr_tobe_checked = {
                    "displaystyle": "false",
                    "mathsize": "normal",
                    "mathvariant": "normal",
                    "fence": "false",
                    "accent": "false",
                    "movablelimits": "false",
                    "largeop": "false",
                    "stretchy": "false",
                    "lquote": "&quot;",
                    "rquote": "&quot;",
                    "overflow": "linebreak",
                    "display": "block",
                    "denomalign": "center",
                    "numalign": "center",
                    "align": "axis",
                    "rowalign": "baseline",
                    "columnalign": "center",
                    "alignmentscope": "true",
                    "equalrows": "true",
                    "equalcolumns": "true",
                    "groupalign": "{left}",
                    "linebreak": "auto",
                    "accentunder": "false",
                }

                mml_mod = attribute_definition(
                    mml_org,
                    elements,
                    attr_tobe_removed,
                    attr_tobe_checked,
                    os.path.join(type_of_folder_path, file_path),
                )

                if verbose:
                    lock.acquire()
                    print(
                        "=============== Printing Modified MML ================"
                    )
                    print("Modified: \n")
                    print(mml_mod)
                    lock.release()

                # Writing modified MML
                tyf_path = os.path.join(simp_mml_folder_path, type_of_folder)
                if not os.path.exists(tyf_path):
                    subprocess.call(["mkdir", tyf_path])

                mod_file_path = os.path.join(tyf_path, file_path)
                with open(mod_file_path, "w") as mml_mod_file_path:
                    mml_mod_file_path.write(mml_mod)


def unicode_logger(mml_code, running_path):

    global lock

    code_dict = {}

    symbol_index = [
        i for i, c in enumerate(mml_code.split()) if ";<!--" in c and "&#x" in c
    ]

    if verbose:
        lock.acquire()
        print(" ===+=== " * 10)
        print("running_path:  ", running_path)
        lock.release()

    if len(symbol_index) != 0:

        for si in symbol_index:

            split_0 = mml_code.split()[si]

            # grabbing part which has '#x<ascii-code>' in it.
            ind = [i for i, c in enumerate(split_0.split(";")) if "#x" in c]
            split_1 = split_0.split(";")[ind[0]]

            code = split_1.split("x")[1]

            ascii_code, unicode = code, mml_code.split()[si + 1]

            lock.acquire()
            unicode_logger.info(f"{running_path} -- {unicode}:{ascii_code}")
            lock.release()

        # Printing final MML code
        if verbose:
            lock.acquire()
            print(f"{running_path} -- {unicode}:{ascii_code}")
            lock.release()


# Removing unnecessary information or attributes having default values
def attribute_definition(
    mml_code, elements, attr_tobe_removed, attr_tobe_checked, running_path
):

    global lock

    # Defining array to keep Attribute definition
    definition_array = []

    for ele in elements:

        # Getting indices of the position of the element in the MML code
        position = [
            i for i in re.finditer(r"\b%s\b" % re.escape(ele), mml_code)
        ]

        for p in position:

            # Attribute begining and ending indices
            (attr_begin, attr_end) = p.span()

            # length of the definition of the attribute
            length = mml_code[attr_end:].find(">")

            if length > 0:

                # Grabbing definition
                definition = mml_code[attr_end : attr_end + length].split()

                # Append unique definition
                for deftn in definition:
                    if deftn not in definition_array:
                        definition_array.append(deftn)

    # remove all the attributes that need to be removed
    for darr in definition_array:
        if "=" in darr:
            # Attribute and its value -- of the element
            attribute_parameter = darr.replace(" ", "").split("=")[0]
            attribute_value = darr.replace(" ", "").split("=")[1]

            # If Attribute has a defualt value, we can remove it
            # Checking which attributes can be removed
            if attribute_parameter not in attr_tobe_removed:
                if attribute_parameter in attr_tobe_checked.keys():
                    if (
                        attribute_value.replace("\\", "").replace('"', "")
                        == attr_tobe_checked[attribute_parameter]
                    ):
                        mml_code = mml_code.replace(darr, "")
            else:
                mml_code = mml_code.replace(darr, "")

    return mml_code


if __name__ == "__main__":

    for year in years:
        main(config, str(year))

    # Printing stoping time
    print(" ")
    stop_time = datetime.now()
    print("Stoping at:  ", stop_time)
    print(" ")
    print("MathML simplification has completed.")
