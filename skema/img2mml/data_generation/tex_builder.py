# CREATING TEX FILES OF THE LATEX EQUATIONS

import os, subprocess, json
import multiprocessing
from datetime import datetime
from multiprocessing import Pool, Lock, TimeoutError

# Printing starting time
print(" ")
start_time = datetime.now()
print("Starting at:  ", start_time)


# Defining global lock
lock = Lock()


def main(config, year):

    src_path = config["source_directory"]
    destination = config["destination_directory"]
    directories = config["months"].split(",")
    verbose = config["verbose"]

    for month_dir in directories:
        month_dir = str(month_dir)
        base_dir = os.path.join(destination, f"{year}/{month_dir}")

        # Latex_equations directory
        latex_equations = os.path.join(base_dir, "latex_equations")

        # tex_files dumping directory
        tex_files = os.path.join(base_dir, "tex_files")
        if not os.path.exists(tex_files):
            subprocess.call(["mkdir", tex_files])

        for folder in os.listdir(latex_equations):
            tex_builder([folder, tex_files, latex_equations])


def tex_builder(args_list):

    # Unpacking argments list
    (folder, tex_files, latex_equations) = args_list

    # creating tex folders for Large and Small equations
    tex_folder = os.path.join(tex_files, folder)
    tex_folder_large_eqn = os.path.join(tex_folder, "Large_eqns")
    tex_folder_small_eqn = os.path.join(tex_folder, "Small_eqns")
    for F in [tex_folder, tex_folder_large_eqn, tex_folder_small_eqn]:
        if not os.path.exists(F):
            subprocess.call(["mkdir", F])

    # reading eqns of paper from folder in latex_equations
    path_to_folder = os.path.join(latex_equations, folder)
    large_eqn_path = os.path.join(path_to_folder, "Large_eqns")
    small_eqn_path = os.path.join(path_to_folder, "Small_eqns")

    # Dealing with "/DeclareMathOperator"
    dmo_file = os.path.join(path_to_folder, "DeclareMathOperator_paper.txt")
    with open(dmo_file, "r") as file:
        dmo = file.readlines()
        file.close()

    # initializing /DeclareMathOperator dictionary
    keyword_dict = {}
    for i in dmo:
        ibegin, iend = i.find("{"), i.find("}")
        keyword_dict[i[ibegin + 1 : iend]] = i

    # Dealing with "Macros"
    macro_file = os.path.join(path_to_folder, "Macros_paper.txt")
    with open(macro_file, "r") as file:
        macro = file.readlines()
        file.close()

    # initializing /Macros dictionary
    keyword_macro_dict = {}
    for i in macro:
        ibegin, iend = i.find("{"), i.find("}")
        keyword_macro_dict[i[ibegin + 1 : iend]] = i

    # eqn_path to the folder containing Large and Small equations
    for eqn_path in [large_eqn_path, small_eqn_path]:
        for eqn_file in os.listdir(eqn_path):

            main_file = os.path.join(eqn_path, eqn_file)

            with open(main_file, "r") as FILE:
                eqn = FILE.readlines()
                FILE.close()

            tex_name = eqn_file.split(".")[0]

            # calling function to create tex doc for the particular folder --> giving all latex eqns, DMOs, Macros and tex_folder path as arguments
            if len(eqn) != 0:
                if eqn_path == large_eqn_path:
                    create_tex_doc(
                        eqn[0],
                        keyword_dict,
                        keyword_macro_dict,
                        tex_folder_large_eqn,
                        tex_name,
                    )
                else:
                    create_tex_doc(
                        eqn[0],
                        keyword_dict,
                        keyword_macro_dict,
                        tex_folder_small_eqn,
                        tex_name,
                    )


# function to create tex documents for each eqn in the folder
def create_tex_doc(eqn, keyword_dict, keyword_macro_dict, tex_folder, tex_name):

    # checking \DeclareMathOperator and Macros
    declare_math_operator_in_eqn = [
        kw for kw in keyword_dict.keys() if kw in eqn
    ]
    macros_in_eqn = [kw for kw in keyword_macro_dict.keys() if kw in eqn]
    preamble_dmo, preamble_macro = "", ""
    for d in declare_math_operator_in_eqn:
        preamble_dmo += "{} \n".format(keyword_dict[d])
    for m in macros_in_eqn:
        preamble_macro += "{} \n".format(keyword_macro_dict[m])

    # creating tex file
    path_to_tex = os.path.join(tex_folder, "{}.tex".format(tex_name))
    with open(path_to_tex, "w") as f_input:
        f_input.write(template(eqn, preamble_dmo, preamble_macro))
        f_input.close()


# Template for the TeX files
def template(eqn, preamble_dmo, preamble_macro):

    # writing tex document for respective eqn
    temp1 = (
        "\\documentclass{standalone}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{amssymb}\n"
    )
    temp2 = (
        "\\begin{document}\n"
        f"$\\displaystyle {{{{ {eqn} }}}} $\n"
        "\\end{document}"
    )

    temp = temp1 + preamble_dmo + preamble_macro + temp2
    return temp


if __name__ == "__main__":
    # read config file
    config_path = "data_generation_config.json"
    with open(config_path, "r") as cfg:
        config = json.load(cfg)

    for year in config["years"].split(","):
        main(config, str(year))

    # Printing stopping time
    print("Stopping at:  ", datetime.now())
