import os
import re
import shutil
import argparse

def process_file(file_path):
    # Extract the file name without extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Create a directory named include_$NAME_OF_FILE
    include_dir = os.path.join(os.path.dirname(file_path), f"include_{file_name}")
    os.makedirs(include_dir, exist_ok=True)

    # Read the file to find #include directives
    with open(file_path, 'r') as file:
        content = file.read()

        # Use regular expression to find #include directives
        include_matches = re.findall(r'#include\s+["<](.*?)[">]', content)

        # Create blank dummy files for each include
        for include_file in include_matches:
            include_file_path = os.path.join(include_dir, *include_file.split('/'))
            os.makedirs(os.path.dirname(include_file_path), exist_ok=True)
            with open(include_file_path, 'w'):
                pass

def process_directory(directory_path):
    # Walk through the directory and process each file
    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            if file_name.endswith(('.c', '.cpp', '.h', '.hpp', ".f", ".F", ".f90", ".F90")):
                file_path = os.path.join(root, file_name)
                process_file(file_path)

def main():
    parser = argparse.ArgumentParser(description='Process C/C++ files and create include directories with dummy files.')
    parser.add_argument('target_directory', metavar='TARGET_DIRECTORY', type=str,
                        help='The path to the target directory containing C/C++ source files')

    args = parser.parse_args()

    # Process the specified directory
    process_directory(args.target_directory)

    print("Script execution completed.")

if __name__ == "__main__":
    main()
