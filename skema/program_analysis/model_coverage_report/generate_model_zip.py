import os
import shutil
import argparse
from zipfile import ZipFile
from tempfile import TemporaryDirectory
from pathlib import Path

from skema.skema_py.server import SUPPORTED_FILE_EXTENSIONS

def filter_and_copy_files(src_dir, zip_file, target_extensions):
    with ZipFile(zip_file, 'w') as zip:
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                file_path = Path(root) / file
                file_extension = file_path.suffix

                if file_extension in target_extensions:
                    arcname = file_path.relative_to(src_dir)
                    zip.write(file_path, arcname=arcname)

def main():
    parser = argparse.ArgumentParser(description="Filter and archive files based on specified extensions.")
    parser.add_argument("src_directory", help="The source directory to process.")
    parser.add_argument("zip_archive", help="The name of the zip archive to create.")
    parser.add_argument("--extensions", nargs="*", default=SUPPORTED_FILE_EXTENSIONS, help="List of target file extensions to keep.")

    args = parser.parse_args()

    filter_and_copy_files(args.src_directory, args.zip_archive, args.extensions)

if __name__ == "__main__":
    main()
