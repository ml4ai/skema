import argparse
import tempfile
import os

from skema.program_analysis.single_file_ingester import process_file
from skema.gromet.fn import GrometFNModuleCollection

def process_snippet(source: str, extension:str, write_to_file=False) -> GrometFNModuleCollection:
    ''' Run a Python or Fortran code snippet through the CODE2FN pipeline and return the GrometFNModuleCollection.
        Optionally, output the Gromet JSON to a file.
    '''
    
    # Create temporary snippet file
    tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=extension)
    tmp.write(source)
    tmp.close()

    gromet_collection = process_file(tmp.name, write_to_file)

    # Delete temporary snippet file
    os.unlink(tmp.name)

    return gromet_collection

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source", type=str, help="Python or Fortran source code"
    )
    parser.add_argument(
        "extension", type=str, help="The language file extension of the code snippet (i.e. .py, .f95)"
    )
    args = parser.parse_args()
    process_snippet(args.source, args.extension, True)
