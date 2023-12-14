from pathlib import Path
import re
import argparse

def process_file(file_path):
    # Extract the file name without extension
    file_name = Path(file_path).stem

    # Create a directory named include_$NAME_OF_FILE
    include_dir = Path(file_path).parent / f"include_{file_name}"
    include_dir.mkdir(parents=True, exist_ok=True)

    # Read the file to find #include directives
    with open(file_path, 'r') as file:
        content = file.read()

        # Use regular expression to find #include directives
        include_matches = re.findall(r'#include\s+["<](.*?)[">]', content)

        # Create blank dummy files for each include
        for include_file in include_matches:
            include_file_path = include_dir / Path(*include_file.split('/'))
            include_file_path.parent.mkdir(parents=True, exist_ok=True)
            include_file_path.touch()

def process_directory(directory_path):
    # Walk through the directory and process each file
    for file_path in Path(directory_path).rglob('*'):
        if file_path.suffix.lower() in {'.c', '.cpp', '.h', '.hpp', '.f', '.F', '.f90', '.F90'}:
            process_file(file_path)

def main():
    parser = argparse.ArgumentParser(description='Process gcc files and create include directories with dummy files.')
    parser.add_argument('target_directory', metavar='TARGET_DIRECTORY', type=str,
                        help='The path to the target directory containing the gcc source files')
    args = parser.parse_args()

    process_directory(args.target_directory)

if __name__ == "__main__":
    main()
