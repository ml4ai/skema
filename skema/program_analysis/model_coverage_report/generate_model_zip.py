import os
import argparse
from zipfile import ZipFile
from pathlib import Path

from skema.skema_py.server import SUPPORTED_FILE_EXTENSIONS

def filter_and_copy_files(src_dir, zip_file, target_extensions):
    """Create zip archive for model, filtering out unsupported file types."""
    with ZipFile(zip_file, 'w') as zip:
        for root, dirs, files in os.walk(src_dir):
            for dir_name in dirs:
                if dir_name.startswith("include_"):
                    dir_path = Path(root) / dir_name
                    arcname = dir_path.relative_to(src_dir)
                    zip.write(dir_path, arcname=arcname)

                    # Iterate over files in the "include_" directory and add them to the zip
                    for subdir_root, subdir_dirs, subdir_files in os.walk(dir_path):
                        for file in subdir_files:
                            file_path = Path(subdir_root) / file
                            arcname = file_path.relative_to(src_dir)
                            zip.write(file_path, arcname=arcname)

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
