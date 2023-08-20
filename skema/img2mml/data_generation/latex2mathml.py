# CONVERT LaTeX EQUATION TO MathML CODE USING MathJax
import time
import requests
import subprocess
import os
import json
import multiprocessing
import logging
from datetime import datetime
from multiprocessing import Pool, Lock, Event
import re
from pathlib import Path

# Printing starting time
print(" ")
start_time = datetime.now()
print("starting at:  ", start_time)

# Defining global lock
lock = Lock()

pause_event = Event()  # suspend and resume processing
equation_counter = multiprocessing.Value("i", 0)
time.sleep(3)
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
num_cpus = config["num_cpus"]


def main(year):
    # Setting up Logger - To get log files
    log_format = "%(levelname)s:%(message)s"
    logfile_dst = os.path.join(destination, f"{year}")
    begin_month, end_month = directories[0], directories[-1]
    logging.basicConfig(
        filename=os.path.join(
            logfile_dst,
            f"logs/{begin_month}_{end_month}_mathjax_mml_newLock.log",
        ),
        level=logging.DEBUG,
        format=log_format,
        filemode="w",
    )

    for month_dir in directories:
        month_dir = str(month_dir)
        print(month_dir)

        root = os.path.join(destination, f"{year}/{month_dir}")

        # Path to image directory
        folder_images = os.path.join(root, "latex_images")

        # Path to directory contain MathML eqns
        mml_dir = os.path.join(root, "mathjax_mml")

        if not os.path.exists(mml_dir):
            subprocess.call(["mkdir", mml_dir])

        os.listdir(mml_dir)

        temp = []

        for folder in os.listdir(folder_images):
            # if folder not in mml_folder_list:

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

        with Pool(num_cpus) as pool:
            pool.map(creating_final_equations, temp)


def creating_macro_dmo_dictionaries(root, folder):
    macro_file = os.path.join(root, f"latex_equations/{folder}/macros.txt")
    with open(macro_file, "r") as file:
        macro = file.readlines()
        file.close()
    keyword_macro_dict = {}
    for i in macro:
        command_pattern = r"\\(new|renew)command\{(.*?)\}\{(.*)\}"
        command_matches = re.findall(command_pattern, i)
        if len(command_matches) > 0:
            if len(command_matches[0]) == 3:
                if command_matches[0][1] != "":
                    keyword_macro_dict[
                        command_matches[0][1]
                    ] = command_matches[0][2]

    # Creating dmo dictionary
    dmo_file = os.path.join(
        root, f"latex_equations/{folder}/declare_math_operator.txt"
    )
    with open(dmo_file, "r") as file:
        dmo = file.readlines()
        file.close()
    keyword_dict = {}
    for i in dmo:
        dmo_pattern = r"\\DeclareMathOperator\{(.*?)\}\{(.*)\}"
        dmo_matches = re.findall(dmo_pattern, i)
        if len(dmo_matches) > 0:
            if len(dmo_matches[0]) == 2:
                if dmo_matches[0][0] != "":
                    keyword_dict[dmo_matches[0][0]] = dmo_matches[0][1]

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
    large_mml = os.path.join(mml_folder, "large_mml")
    small_mml = os.path.join(mml_folder, "small_mml")
    for F in [mml_folder, large_mml, small_mml]:
        if not os.path.exists(F):
            subprocess.call(["mkdir", F])

    # Appending all the eqns of the folder/paper to Latex_strs_json
    # along with their respective macros and Declare Math Operator commands.

    # Creating array of final eqns
    large_eqns = os.path.join(folder_images, f"{folder}/large_eqns")
    small_eqns = os.path.join(folder_images, f"{folder}/small_eqns")

    for type_of_folder in [large_eqns, small_eqns]:
        for index, eqn in enumerate(os.listdir(type_of_folder)):
            if ".png" in eqn:
                try:
                    file_name = eqn.split("-")[0].split(".")[0]

                    eqnstype = (
                        "large_eqns"
                        if type_of_folder == large_eqns
                        else "small_eqns"
                    )
                    file_path = os.path.join(
                        root,
                        f"latex_equations/{folder}/{eqnstype}/{file_name}.txt",
                    )

                    final_eqn = ""

                    text_eqns = open(file_path, "r").readlines()
                    if len(text_eqns) != 0:
                        if len(text_eqns) > 1:
                            text_eqn = " "
                            for e in text_eqns:
                                text_eqn = text_eqn + e.replace("\n", "")
                        else:
                            text_eqn = text_eqns[0]

                    for key, value in keyword_macro_dict.items():
                        text_eqn = text_eqn.replace(key, value)

                    for key, value in keyword_dict.items():
                        text_eqn = text_eqn.replace(key, value)

                    string = text_eqn

                    # Correcting keywords written in an incorrect way
                    for sub in string.split(" "):
                        if "cong" in sub:
                            sub = sub.replace("\\cong", "{\\cong}")

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

                    if not os.path.exists(mml + f"/{file_name}.xml"):
                        mjxmml(
                            file_name, folder, final_eqn, type_of_folder, mml
                        )

                except:
                    lock.acquire()
                    if verbose:
                        print(" ")
                        print(
                            f" {type_of_folder}/{file_name}: can not be converted."
                        )
                        print(
                            " ==================================================== "
                        )

                    logger.warning(
                        f"{type_of_folder}/{file_name}: can not be converted."
                    )
                    lock.release()


def correct_phi(string):
    pattern = r"(<mi>&#x03C6;<!-- φ --></mi> <mi>&#x03C6;<!-- φ --></mi> <mtext>&#xA0;</mtext> <mi mathvariant=\"normal\">&#x0393;<!-- Γ --></mi> <mo stretchy=\"false\">[</mo> <mi>f</mi> <mo stretchy=\"false\">(</mo> <mi>t</mi> <mo stretchy=\"false\">)</mo> <mi>cos</mi> <mo>&#x2061;<!-- ⁡ --></mo> <mi>&#x03C6;<!-- φ --></mi> )(.+?)( <mo stretchy=\"false\">]</mo>)"
    replacement_dict = {
        "\\": "place_holder1",
        "[": "place_holder2",
        "]": "place_holder3",
        ">(<": "place_holder4",
        ">)<": "place_holder5",
    }
    for key, val in replacement_dict.items():
        pattern = pattern.replace(key, val)
        string = string.replace(key, val)

    matches = re.findall(pattern, string)
    for match in matches:
        placeholder = match[1]
        string = string.replace(
            "".join(match), f"{placeholder} <mi>&#x0278;<!-- ɸ --></mi>"
        )

    for key, val in replacement_dict.items():
        string = string.replace(val, key)
    return string


def restart_mathjax_server():
    requests.get("http://localhost:8081/restart")


def mjxmml(file_name, folder, eqn, type_of_folder, mml_path):
    global lock, pause_event
    lock.acquire()
    print(mml_path + f"/{file_name}.xml")
    # Open the file for reading
    with open("statistical_results.txt", "r") as f:
        # Read the first line and split it into a list of strings
        numbers = f.readline().strip().split(", ")
        # Convert the strings to integers
        numbers = [int(num) for num in numbers]
        # Update the total number by adding 1
        numbers[0] += 1

    # Open the file for writing
    with open("statistical_results.txt", "w") as f:
        # Write the updated numbers to the file
        f.write(", ".join(str(num) for num in numbers))

    lock.release()
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
            f"converting latex to MML using MathJax webserver {file_name}...."
        )
        print(" ")
        print(f"response of the webservice request: {res.text}")
        lock.release()

    # Capturing the keywords not supported by MathJax
    if "FAILED" in res.content.decode("utf-8"):
        lock.acquire()
        # Open the file for reading
        with open("statistical_results.txt", "r") as f:
            # Read the first line and split it into a list of strings
            numbers = f.readline().strip().split(", ")
            # Convert the strings to integers
            numbers = [int(num) for num in numbers]
            # Update the failed number by adding 1
            numbers[1] += 1

        # Open the file for writing
        with open("statistical_results.txt", "w") as f:
            # Write the updated numbers to the file
            f.write(", ".join(str(num) for num in numbers))

        lock.release()
        # Just to check errors
        tex_parse_error = res.content.decode("utf-8").split("::")[1]

        # Logging incorrect/ unsupported keywords along with their equations
        if "Undefined control sequence" in tex_parse_error:
            unsupported_keyword = tex_parse_error.split("\\")[-1]

            lock.acquire()
            if verbose:
                print(
                    f"{type_of_folder}/{file_name}:{unsupported_keyword} is either not \
                        supported by MathJax or incorrectly written."
                )

            logger.warning(
                f"{type_of_folder}/{file_name}:{unsupported_keyword} is either not \
                    supported by MathJax or incorrectly written."
            )
            lock.release()

        elif (
            "TypeError: Cannot read property 'root' of undefined"
            in tex_parse_error
        ):
            lock.acquire()
            print(folder)
            logger.warning(
                f"Math Processing Error: Maximum call stack size exceeded.\
                     Killing the process and server. \
                    {type_of_folder}/{file_name}:{tex_parse_error}"
            )
            lock.release()
            pause_event.clear()
            restart_mathjax_server()
            time.sleep(3)
            pause_event.set()

        # Logging errors other than unsupported keywords
        else:
            lock.acquire()
            if verbose:
                print(
                    f"{type_of_folder}/{file_name}:{tex_parse_error} is an \
                        error produced by MathJax webserver."
                )

            logger.warning(
                f"{type_of_folder}/{file_name}:{tex_parse_error} is an \
                    error produced by MathJax webserver."
            )
            lock.release()

    else:
        # Cleaning and Dumping the MathML strings to JSON file
        mml = cleaning_mml(res.text)
        # Replacing the wrong generation from MathJax
        mml = re.sub("\s+", " ", mml)
        mml = mml.replace(
            r"<msub> <mi>&#x2113;<!-- ℓ --></mi> <mn>1</mn> </msub> <mo>,</mo> <msub> <mi>&#x2113;<!-- ℓ --></mi> <mn>2</mn> </msub>",
            "",
        )
        mml = mml.replace(
            r"<mi>&#x03C6;<!-- φ --></mi> <mi>&#x03C6;<!-- φ --></mi> <mtext>&#xA0;</mtext> <mi mathvariant=\"normal\">&#x0393;<!-- Γ --></mi> <mo stretchy=\"false\">[</mo> <mi>f</mi> <mo stretchy=\"false\">(</mo> <mi>t</mi> <mo stretchy=\"false\">)</mo> <mi>cos</mi> <mo>&#x2061;<!-- ⁡ --></mo> <mi>&#x03C6;<!-- φ --></mi> <mo stretchy=\"false\">]</mo>",
            r"<mi>&#x0278;<!-- ɸ --></mi>",
        )
        mml = correct_phi(mml)

        if verbose:
            lock.acquire()
            print(f"writing {file_name}")
            lock.release()

        with open(
            os.path.join(mml_path, f"{file_name}.xml"), "w"
        ) as mml_output:
            mml_output.write(mml)
            mml_output.close()

        if "\\boldsymbol" in eqn or "\\mathbf" in eqn or "\\bf" in eqn:
            lock.acquire()
            # Open the file for reading
            with open("statistical_results.txt", "r") as f:
                # Read the first line and split it into a list of strings
                numbers = f.readline().strip().split(", ")
                # Convert the strings to integers
                numbers = [int(num) for num in numbers]
                # Update the boldface number by adding 1
                numbers[2] += 1

            # Open the file for writing
            with open("statistical_results.txt", "w") as f:
                # Write the updated numbers to the file
                f.write(", ".join(str(num) for num in numbers))

            # Open the file for writing
            with open("boldface_list.txt", "a") as f:
                # Write the updated numbers to the file
                f.write(os.path.join(mml_path, f"{file_name}.xml") + "\n")

            lock.release()

    lock.acquire()
    equation_counter.value += 1

    # To avoid the buffer issue from MathJax, 
    # restart the service once processing 1000 equations
    if equation_counter.value % 1000 == 0:
        pause_event.clear()
        restart_mathjax_server()
        time.sleep(3)
        pause_event.set()
        equation_counter.value = 0
    else:
        pause_event.set()

    lock.release()
    pause_event.wait()


def cleaning_mml(res):
    # Removing "\ and /" at the begining and at the end
    res = res[res.find("<") :]
    res = res[::-1][res[::-1].find(">") :]
    res = res[::-1]

    # Removing "\\n"
    res = res.replace(">\\n", ">")
    return res


if __name__ == "__main__":
    current_dir = Path(os.getcwd())

    statistical_file = current_dir / "statistical_results.txt"
    if not statistical_file.exists():
        with open(statistical_file, "w") as f:
            f.write("0, 0, 0\n")  # total, failed, boldface

    boldface_files = current_dir / "boldface_list.txt"
    if not boldface_files.exists():
        boldface_files.touch()

    for year in years:
        year = year.strip()
        print(year)
        main(str(year))

    # Printing stopping time
    print(" ")
    stop_time = datetime.now()
    print("stoping at:  ", stop_time)
    print(" ")
    print("LaTeX-MathML conversion has completed.")
