# CONVERT LaTeX EQUATION TO MathML CODE USING MathJax
import requests
import subprocess, os
import json
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

# defining logger
logger = logging.getLogger()

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
    log_format = "%(levelname)s:%(message)s"
    logfile_dst = os.path.join(destination, f"{year}")
    begin_month, end_month = directories[0], directories[-1]
    logging.basicConfig(
        filename=os.path.join(
            logfile_dst, f"{begin_month}_{end_month}_MathJax_MML_newLock.log"
        ),
        level=logging.DEBUG,
        format=log_format,
        filemode="w",
    )

    for month_dir in directories:

        month_dir = str(month_dir)
        root = os.path.join(destination, f"{year}/{month_dir}")

        # Path to image directory
        folder_images = os.path.join(root, "latex_images")

        # Path to directory contain MathML eqns
        mml_dir = os.path.join(root, "Mathjax_mml")

        if not os.path.exists(mml_dir):
            subprocess.call(["mkdir", mml_dir])

        mml_folder_list = os.listdir(mml_dir)

        temp = []

        for folder in os.listdir(folder_images):

            if folder not in mml_folder_list:

                # Creating macros dictionary
                (
                    keyword_macro_dict,
                    keyword_dict,
                ) = creating_macro_dmo_dictionaries(root, folder)
                temp.append(
                    [
                        month_dir,
                        root,
                        mml_dir,
                        folder_images,
                        folder,
                        keyword_macro_dict,
                        keyword_dict,
                    ]
                )

        with Pool(multiprocessing.cpu_count() - 10) as pool:
            result = pool.map(creating_final_equations, temp)


def creating_macro_dmo_dictionaries(root, folder):

    macro_file = os.path.join(
        root, f"latex_equations/{folder}/Macros_paper.txt"
    )
    with open(macro_file, "r") as file:
        macro = file.readlines()
        file.close()
    keyword_macro_dict = {}
    for i in macro:
        ibegin, iend = i.find("{"), i.find("}")
        keyword_macro_dict[i[ibegin + 1 : iend]] = i

    # Creating dmo dictionary
    dmo_file = os.path.join(
        root, f"latex_equations/{folder}/DeclareMathOperator_paper.txt"
    )
    with open(dmo_file, "r") as file:
        dmo = file.readlines()
        file.close()
    keyword_dict = {}
    for i in dmo:
        ibegin, iend = i.find("{"), i.find("}")
        keyword_dict[i[ibegin + 1 : iend]] = i

    return (keyword_macro_dict, keyword_dict)


def creating_final_equations(args_list):

    global lock

    # Unpacking the args_list
    (
        month_dir,
        root,
        mml_dir,
        folder_images,
        folder,
        keyword_macro_dict,
        keyword_dict,
    ) = args_list

    # Creating folder for MathML codes for specific file
    mml_folder = os.path.join(mml_dir, folder)

    # Creating folder for Large and Small eqns
    large_mml = os.path.join(mml_folder, "Large_MML")
    small_mml = os.path.join(mml_folder, "Small_MML")
    for F in [mml_folder, large_mml, small_mml]:
        if not os.path.exists(F):
            subprocess.call(["mkdir", F])

    # Appending all the eqns of the folder/paper to Latex_strs_json
    # along with their respective macros and Declare Math Operator commands.

    # Creating array of final eqns
    large_eqns = os.path.join(folder_images, f"{folder}/Large_eqns")
    small_eqns = os.path.join(folder_images, f"{folder}/Small_eqns")

    for type_of_folder in [large_eqns, small_eqns]:

        for index, eqn in enumerate(os.listdir(type_of_folder)):

            if ".png" in eqn:

                try:
                    file_name = eqn.split("-")[0].split(".")[0]

                    eqnstype = (
                        "Large_eqns"
                        if type_of_folder == large_eqns
                        else "Small_eqns"
                    )
                    file_path = os.path.join(
                        root,
                        f"latex_equations/{folder}/{eqnstype}/{file_name}.txt",
                    )

                    final_eqn = ""

                    text_eqn = open(file_path, "r").readlines()[0]
                    macros_in_eqn = [
                        kw
                        for kw in keyword_macro_dict.keys()
                        if kw in text_eqn
                    ]
                    dmos_in_eqn = [
                        kw for kw in keyword_dict.keys() if kw in text_eqn
                    ]

                    # Writing macros, dmos, and text_eqn as one string
                    MiE, DiE = "", ""
                    for macro in macros_in_eqn:
                        MiE = MiE + keyword_macro_dict[macro] + " "
                    for dmo in dmos_in_eqn:
                        DiE = DiE + keyword_dict[dmo] + " "

                    string = MiE + DiE + text_eqn

                    # Removing unsupported keywords
                    for tr in [
                        "\\ensuremath",
                        "\\xspace",
                        "\\aligned",
                        "\\endaligned",
                        "\\span",
                    ]:
                        string = string.replace(tr, "")

                    # Correcting keywords written in an incorrect way
                    for sub in string.split(" "):
                        if "cong" in sub:
                            sub = sub.replace("\\cong", "{\\cong}")
                        if "mathbb" in sub:
                            if sub[sub.find("\\mathbb") + 7] != "{":
                                mathbb_parameter = sub[
                                    sub.find("\\newcommand")
                                    + 12 : sub.find("}")
                                ].replace("\\", "")
                                sub = (
                                    sub[: sub.find("\\mathbb") + 7]
                                    + "{"
                                    + mathbb_parameter
                                    + "}"
                                    + sub[
                                        sub.find("\\mathbb")
                                        + 7
                                        + len(mathbb_parameter) :
                                    ]
                                )
                        if "mathbf" in sub:
                            if sub[sub.find("\\mathbf") + 7] != "{":
                                mathbf_parameter = sub[
                                    sub.find("\\newcommand")
                                    + 12 : sub.find("}")
                                ].replace("\\", "")
                                sub = (
                                    sub[: sub.find("\\mathbf") + 7]
                                    + "{"
                                    + mathbf_parameter
                                    + "}"
                                    + sub[
                                        sub.find("\\mathbf")
                                        + 7
                                        + len(mathbf_parameter) :
                                    ]
                                )

                        final_eqn += sub + " "

                    # Printing the final equation string
                    if verbose:
                        lock.acquire()
                        print("final equation is  ", final_eqn)
                        lock.release()

                    mml = (
                        large_mml
                        if type_of_folder == large_eqns
                        else small_mml
                    )

                    mjxmml(file_name, folder, final_eqn, type_of_folder, mml)

                except:
                    print("passing to except....")
                    lock.acquire()
                    if verbose:
                        print(" ")
                        print(
                            f" {type_of_folder}/{file_name}: can not be converted."
                        )
                        print(
                            " =============================================================== "
                        )

                    logger.warning(
                        f"{type_of_folder}/{file_name}: can not be converted."
                    )
                    lock.release()


def mjxmml(file_name, folder, eqn, type_of_folder, mml_path):

    global lock

    print(eqn)


    # Define the webservice address
    webservice = "http://localhost:8081"
    # Load the LaTeX string data
    # Translate and save each LaTeX string using the NodeJS service for MathJax
    res = requests.post(
        f"{webservice}/tex2mml",
        headers={"Content-type": "application/json"},
        json={"tex_src": json.dumps(eqn)},
    )

    if verbose:
        lock.acquire()
        print(
            f"Converting latex equation to MathML using MathJax webserver of {file_name}...."
        )
        print(" ")
        print(f"Response of the webservice request: {res.text}")
        lock.release()

    # Capturing the keywords not supported by MathJax
    if "FAILED" in res.content.decode("utf-8"):
        # Just to check errors
        tex_parse_error = res.content.decode("utf-8").split("::")[1]

        # Logging incorrect/ unsupported keywords along with their equations
        if "Undefined control sequence" in tex_parse_error:
            unsupported_keyword = tex_parse_error.split("\\")[-1]

            lock.acquire()
            if verbose:
                print(
                    f"{type_of_folder}/{file_name}:{unsupported_keyword} is either not supported by MathJax or incorrectly written."
                )

            logger.warning(
                f"{type_of_folder}/{file_name}:{unsupported_keyword} is either not supported by MathJax or incorrectly written."
            )
            lock.release()

        elif (
            "TypeError: Cannot read property 'root' of undefined"
            in tex_parse_error
        ):
            lock.acquire()
            print(folder)
            logger.warning(
                f"{type_of_folder}/{file_name}:{tex_parse_error} -- Math Processing Error: Maximum call stack size exceeded. Killing the process and server."
            )
            lock.release()

        # Logging errors other than unsupported keywords
        else:
            lock.acquire()
            if verbose:
                print(
                    f"{type_of_folder}/{file_name}:{tex_parse_error} is an error produced by MathJax webserver."
                )

            logger.warning(
                f"{type_of_folder}/{file_name}:{tex_parse_error} is an error produced by MathJax webserver."
            )
            lock.release()

    else:
        # Cleaning and Dumping the MathML strings to JSON file
        mml = cleaning_mml(res.text)
        print(mml)

        if verbose:
            lock.acquire()
            print(f"writing {file_name}")
            lock.release()

        with open(
            os.path.join(mml_path, f"{file_name}.txt"), "w"
        ) as mml_output:
            mml_output.write(mml)
            mml_output.close()


def cleaning_mml(res):

    # Removing "\ and /" at the begining and at the end
    res = res[res.find("<") :]
    res = res[::-1][res[::-1].find(">") :]
    res = res[::-1]

    # Removing "\\n"
    res = res.replace(">\\n", ">")
    return res


if __name__ == "__main__":

    for year in years:
        main(str(year))

    # Printing stoping time
    print(" ")
    stop_time = datetime.now()
    print("Stoping at:  ", stop_time)
    print(" ")
    print("LaTeX-MathML conversion has completed.")
