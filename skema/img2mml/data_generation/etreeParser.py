import xml.etree.ElementTree as ET
import subprocess, os, json
import sys
import xml.dom.minidom
import multiprocessing
import logging
from xml.etree.ElementTree import ElementTree
from datetime import datetime
from multiprocessing import Pool, Lock, TimeoutError


# Defining global lock
lock = Lock()

# Printing starting time
print(" ")
start_time = datetime.now()
print("starting at:  ", start_time)

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
    log_format = "%(message)s"
    logfile_dst = os.path.join(destination, f"{year}")
    begin_month, end_month = directories[0], directories[-1]
    logging.basicConfig(
        filename=os.path.join(
            logfile_dst, f"{begin_month}-{end_month}_etree.log"
        ),
        level=logging.DEBUG,
        format=log_format,
        filemode="w",
    )

    logger = logging.getLogger()

    # Creating 'etree' directory
    for month_dir in directories:
        month_dir = str(month_dir)
        etreePath = f"{destination}/{year}/{month_dir}/etree"
        sample_etreePath = f"{destination}/{year}/{month_dir}/sample_etree"

        for path in [etreePath, sample_etreePath]:
            if not os.path.exists(path):
                subprocess.call(["mkdir", path])

    for month_dir in directories:
        month_dir = str(month_dir)
        simp_Mathml_path = f"{destination}/{year}/{month_dir}/mathjax_mml"

        args_array = pooling(month_dir, simp_Mathml_path, destination, year)

        with Pool(num_cpus) as pool:
            result = pool.map(etree, args_array)


def pooling(month_dir, simp_Mathml_path, destination, year):

    temp = []

    for subdir in os.listdir(simp_Mathml_path):

        subdir_path = os.path.join(simp_Mathml_path, subdir)
        temp.append([month_dir, subdir, subdir_path, destination, year])

    return temp


def etree(args_array):

    global lock

    # Unpacking the args array
    (month_dir, subdir, subdir_path, destination, year) = args_array

    etree_path = f"{destination}/{year}/{month_dir}/etree"
    sample_etree_path = f"{destination}/{year}/{month_dir}/sample_etree"

    for tyf in ["large_mml", "small_mml"]:

        tyf_path = os.path.join(subdir_path, tyf)

        # create folders
        create_folders(subdir, tyf, etree_path, sample_etree_path)

        for mml_file in os.listdir(tyf_path):

            try:
                mml_file_path = os.path.join(tyf_path, mml_file)

                mml_file_name = mml_file.split(".")[0]

                # converting text file to xml formatted file
                tree1 = ElementTree()
                tree1.parse(mml_file_path)
                sample_path = f"{destination}/{year}/{month_dir}/sample_etree/{subdir}/{tyf}/{mml_file_name}_sample.xml"

                # writing the sample files that will be used to render etree
                tree1.write(sample_path)

                # Writing etree for the xml files
                tree = ET.parse(sample_path)
                xml_data = tree.getroot()
                xmlstr = ET.tostring(xml_data, encoding="utf-8", method="xml")
                input_string = xml.dom.minidom.parseString(xmlstr)
                xmlstr = input_string.toprettyxml()
                xmlstr = os.linesep.join(
                    [s for s in xmlstr.splitlines() if s.strip()]
                )

                result_path = os.path.join(
                    etree_path, f"{subdir}/{tyf}/{mml_file_name}.xml"
                )

                print(xmlstr.encode(sys.stdout.encoding, errors="replace"))
                
                with open(result_path, "wb") as file_out:
                    file_out.write(
                        xmlstr.encode(sys.stdout.encoding, errors="replace")
                    )

            except:
                lock.acquire()
                logger.warning(f"{mml_file_path} not working.")
                lock.release()


def create_folders(subdir, tyf, etree_path, sample_etree_path):

    global lock

    etree_subdir_path = os.path.join(etree_path, subdir)
    etree_tyf_path = os.path.join(etree_subdir_path, tyf)

    sample_etree_subdir_path = os.path.join(sample_etree_path, subdir)
    sample_etree_tyf_path = os.path.join(sample_etree_subdir_path, tyf)

    for F in [
        etree_subdir_path,
        etree_tyf_path,
        sample_etree_subdir_path,
        sample_etree_tyf_path,
    ]:
        if not os.path.exists(F):
            subprocess.call(["mkdir", F])


if __name__ == "__main__":
    for year in years:
        main(str(year))

    # removing sample_trees folder that we was temporarily created while parsing etrees
    os.rmdir(f"{destination}/{year}/{month_dir}/sample_etree")

    # Printing stoping time
    print(" ")
    stop_time = datetime.now()
    print("stopping at:  ", stop_time)
    print(" ")
    print("etree writing process -- completed.")
