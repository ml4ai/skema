import argparse
import tempfile
import os
from pathlib import Path

from skema.program_analysis.multi_file_ingester import process_file_system
from skema.gromet.fn import GrometFNModuleCollection


def process_file(path: str, write_to_file=False, original_source=False) -> GrometFNModuleCollection:
    """Run a single Python or Fortran file through the CODE2FN pipeline and return the GrometFNModuleCollection.
    Optionally, output the Gromet JSON to a file.
    Optionally, include the entire original source code of the file in the GrometFNModuleCollection.
    """

    path_obj = Path(path)
    system_name = path_obj.stem
    file_name = path_obj.name
    root_path = str(path_obj.parent)

    # Create temporary system_filepaths file
    tmp = tempfile.NamedTemporaryFile(mode="w", delete=False)
    tmp.write(file_name)
    tmp.close()

    gromet_collection = process_file_system(
        system_name, root_path, tmp.name, write_to_file, original_source
    )

    # Delete temporary system_filepaths file
    os.unlink(tmp.name)

    return gromet_collection


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=str, help="The relative or absolute path of the file to process"
    )
    parser.add_argument(
        "--source", action="store_true", help="Toggle whether or not to include the full source code of the code in the GroMEt metadata"
    )
    args = parser.parse_args()
    process_file(args.path, True, args.source)
