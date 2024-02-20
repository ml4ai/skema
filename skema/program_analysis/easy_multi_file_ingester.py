import argparse
import tempfile
import os
from pathlib import Path

from skema.program_analysis.multi_file_ingester import process_file_system
from skema.gromet.fn import GrometFNModuleCollection
from skema.skema_py.server import SUPPORTED_FILE_EXTENSIONS

def easy_process_file_system(system_name: str, root_path: str, write_to_file=False, original_source=False, dependency_depth=0) -> GrometFNModuleCollection:
    """
    Like process_file_system but doesn't require a system_filepaths.txt file. Returns a GrometFNModuleCollection.
    Optionally, specify the dependency depth for analysis.

    Parameters:
    system_name (str): The name of the system to ingest.
    root_path (str): The relative or absolute path to the directory to ingest
    write_to_file (bool, optional): Whether or not to output Gromet to file.
    original_source (bool, optional): Whether or not to include original source code in output Gromet.
    dependency_depth (int, optional): Specify the dependency depth for analysis.
    """
    path_obj = Path(root_path).resolve()
    
    # Create temporary system_filepaths file by recursively iterating over all files in root path
    file_paths = []
    for extension in SUPPORTED_FILE_EXTENSIONS:
        file_paths.extend([str(file.relative_to(path_obj)) for file in path_obj.rglob(f"*{extension}")])
    

    tmp = tempfile.NamedTemporaryFile(mode="w", delete=False)
    tmp.write("\n".join(file_paths))
    tmp.close()

    gromet_collection = process_file_system(
        system_name, str(path_obj), tmp.name, write_to_file, original_source, dependency_depth
    )

    # Delete temporary system_filepaths file
    os.unlink(tmp.name)

    return gromet_collection

def generate_statistics():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "system_name", type=str, help="The name of the system to ingest."
    )
    parser.add_argument(
        "root_path", type=str, help="The relative or absolute path to the directory to ingest"
    )
    parser.add_argument(
        "--source", action="store_true", help="Toggle whether or not to include the full source code of the code in the GroMEt metadata"
    )
    parser.add_argument(
        "--dependency_depth", type=int, default=0, help="Specify the dependency depth for analysis"
    )
    args = parser.parse_args()
    easy_process_file_system(args.system_name, args.root_path, True, args.source, args.dependency_depth)
