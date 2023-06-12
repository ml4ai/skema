import argparse
from pathlib import Path
from datetime import datetime 
import os

from skema.program_analysis.single_file_ingester import process_file
from skema.utils.fold import del_nulls, dictionary_to_gromet_json

SUPPORTED_FILE_EXTENSIONS = set([".py", ".f", ".f95"])

def regenerate_examples_google_drive(root_dir: str, gromet_version: str, overwrite=False):
    '''Script to regenerate Gromet FN JSON for Python/Fortran source examples following the Google Drive structure.'''
    '''
    The directory structure looks like:
    root_dir
        example_1
            gromet
                v0.1.6
                    example1.json
                v0.1.5
                    example1.json
            example1.py
    '''
    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        print("ERROR: root_dir argument must point to a directory")
        exit(1)

    for path in root_dir.iterdir():
        if path.isdir():
            output_dir = Path(path, gromet_version)
            gromet_file = Path(output_dir, f"{path.stem}--Gromet-FN-auto.json")
            
            if not overwrite and gromet_file.exists():
                print(f"WARNING: {str(gromet_file)} already exists and overwrite set to False")
                continue

            for file in path.iterdir():
                if file.suffix in SUPPORTED_FILE_EXTENSIONS:
                    gromet_collection = process_file(str(file))
                    print(str(gromet_file))
                    with open(gromet_file, "w") as f:
                        f.write(dictionary_to_gromet_json(del_nulls(gromet_collection.to_dict())))

def regenerate_examples_simple(root_dir: str, output_dir=None, overwrite=False) -> None:
    '''Script to regenerate Gromet FN JSON for Python/Fortran source examples where are source files are in a single directory.'''
    '''
    The directory structure should look like:
    root_dir
        example_1.py
        example_2.py
    '''
    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        print("ERROR: root_dir argument must point to a directory")
        exit(1)

    if output_dir:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            Path.mkdir(output_dir)


    for path in root_dir.iterdir():
        print(str(path))
        if path.suffix.lower() in SUPPORTED_FILE_EXTENSIONS:
            
            if output_dir:
                gromet_file = Path(output_dir, f"{path.stem}--Gromet-FN-auto.json")
            else:
                gromet_file = Path(root_dir, f"{path.stem}--Gromet-FN-auto.json")
                
            if not overwrite and gromet_file.exists():
                print(f"WARNING: {str(gromet_file)} already exists and overwrite set to False")
                continue

            gromet_collection = process_file(str(path))
            with open(gromet_file, "w") as f:
                f.write(dictionary_to_gromet_json(del_nulls(gromet_collection.to_dict())))
        else:
            print(f"WARNING: The file type of {str(path)} is not supported by CODE2FN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="mode", dest="mode", help="Select mode")

    # Subparser for regenerate_examples_google_drive
    parser_google_drive = subparsers.add_parser("google_drive", help="Regenerate examples following Google Drive structure")
    parser_google_drive.add_argument("root_dir", type=str, help="Path to the root directory")
    parser_google_drive.add_argument("gromet_version", type=str, help="Gromet version number")
    parser_google_drive.add_argument("--overwrite", action="store_true", help="Overwrite existing Gromet files")

    # Subparser for regenerate_examples_simple
    parser_simple = subparsers.add_parser("simple", help="Regenerate examples in a single directory")
    parser_simple.add_argument("root_dir", type=str, help="Path to the root directory")
    parser_simple.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser_simple.add_argument("--overwrite", action="store_true", help="Overwrite existing Gromet files")

    args = parser.parse_args()

    if args.mode == "google_drive":
        regenerate_examples_google_drive(args.root_dir, args.gromet_version, args.overwrite)
    elif args.mode == "simple":
        regenerate_examples_simple(args.root_dir, args.output_dir, args.overwrite)
    else:
        parser.print_help()
