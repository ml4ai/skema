
import xml.etree.ElementTree as ET
import subprocess, os
import sys
import xml.dom.minidom
import multiprocessing
import logging
from xml.etree.ElementTree import ElementTree
from datetime import datetime
from multiprocessing import Pool, Lock, TimeoutError


# Defining global lock
lock = Lock()


# Setting up Logger - To get log files
Log_Format = '%(message)s'
logFile_dst = os.path.join(args.source, f'{str(args.year)}/Logs')
begin_month, end_month = str(args.directories[0]), str(args.directories[-1])
logging.basicConfig(filename = os.path.join(logFile_dst, f'{begin_month}-{end_month}_etree.log'),
                    level = logging.DEBUG,
                    format = Log_Format,
                    filemode = 'w')

logger = logging.getLogger()

# Printing starting time
print(' ')
start_time = datetime.now()
print('Starting at:  ', start_time)


# Defining lock
lock = Lock()

def main(config, year):

    src_path = config["source_directory"]
    destination = config["destination_directory"]
    directories = list(config["month"])
    verbose = config["verbose"]

    # Creating 'etree' directory
    for DIR in directories:
        DIR = str(DIR)
        etreePath = f'{destination}/{year}/{DIR}/etree'
        sample_etreePath = f'{destination}/{year}/{DIR}/sample_etree'

        for path in [etreePath, sample_etreePath]:
            if not os.path.exists(path):
                subprocess.call(['mkdir', path])


    for DIR in directories:
        DIR = str(DIR)
        Simp_Mathml_path = f'{destination}/{year}/{DIR}/Simplified_mml'

        args_array = pooling(DIR, Simp_Mathml_path,
                                destination, year)

        with Pool(multiprocessing.cpu_count()-10) as pool:
            result = pool.map(etree, args_array)


def pooling(DIR, Simp_Mathml_path, destination, year):

    temp = []

    for subDIR in os.listdir(Simp_Mathml_path):

        subDIR_path = os.path.join(Simp_Mathml_path, subDIR)
        temp.append([DIR, subDIR, subDIR_path, destination, year])

    return temp

def etree(args_array):

    global lock

    # Unpacking the args array
    (DIR, subDIR, subDIR_path, destination, year) = args_array

    etree_path = f'{destination}/{year}/{DIR}/etree'
    sample_etree_path = f'{destination}/{year}/{DIR}/sample_etree'

    for tyf in ['Large_MML', 'Small_MML']:

        tyf_path = os.path.join(subDIR_path, tyf)

        # create folders
        Create_Folders(subDIR, tyf, etree_path, sample_etree_path)

        for FILE in os.listdir(tyf_path):

            try:
                FILE_path = os.path.join(tyf_path, FILE)

                FILE_name = FILE.split('.')[0]

                # converting text file to xml formatted file
                tree1 = ElementTree()
                tree1.parse(FILE_path)
                sample_path = f'{destination}/{year}/{DIR}/sample_etree/{subDIR}/{tyf}/{FILE_name}_sample.xml'

                # writing the sample files that will be used to render etree
                tree1.write(sample_path)


                # Writing etree for the xml files
                tree = ET.parse(sample_path)
                xml_data = tree.getroot()
                xmlstr = ET.tostring(xml_data, encoding='utf-8', method='xml')
                input_string=xml.dom.minidom.parseString(xmlstr)
                xmlstr = input_string.toprettyxml()
                xmlstr= os.linesep.join([s for s in xmlstr.splitlines() if s.strip()])

                result_path = os.path.join(etree_path, f'{subDIR}/{tyf}/{FILE_name}.xml')

                with open(result_path, "wb") as file_out:
                    file_out.write(xmlstr.encode(sys.stdout.encoding, errors='replace'))

            except:
                lock.acquire()
                logger.warning(f'{FILE_path} not working.')
                lock.release()


def Create_Folders(subDIR, tyf, etree_path, sample_etree_path):

    global lock

    etree_subDIR_path = os.path.join(etree_path, subDIR)
    etree_tyf_path = os.path.join(etree_subDIR_path, tyf)

    sample_etree_subDIR_path = os.path.join(sample_etree_path, subDIR)
    sample_etree_tyf_path = os.path.join(sample_etree_subDIR_path, tyf)

    for F in [etree_subDIR_path, etree_tyf_path, sample_etree_subDIR_path, sample_etree_tyf_path]:
        if not os.path.exists(F):
            subprocess.call(['mkdir', F])


if __name__ == "__main__":
    # read config file
    config_path = "data_generation_config.json"
    with open(config_path, "r") as cfg:
        config = json.load(cfg)

    for year in list(config["years"]):
        main(config, str(year))


    # Printing stoping time
    print(' ')
    stop_time = datetime.now()
    print('Stoping at:  ', stop_time)
    print(' ')
    print('etree writing process -- completed.')
