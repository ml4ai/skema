import subprocess
import tempfile
import os
import argparse
import requests
import tarfile
from pathlib import Path

from skema.gromet.fn import GrometFNModuleCollection
from skema.program_analysis.easy_multi_file_ingester import easy_process_file_system

def process_git_repo(repo_url: str, checkout_ref=None, write_to_file=False, original_source=False, dependency_depth=0) -> GrometFNModuleCollection:
    """
    Clones a Git repository to a temporary directory and ingests it into a GrometFNModuleCollection with an optional dependency depth.
    """
    system_name = Path(repo_url).stem
 
    with tempfile.TemporaryDirectory() as temp:
        cloned_path = Path(temp) / system_name
        try:
            subprocess.check_call(['git', 'clone', repo_url, cloned_path.name], cwd=temp)
            if checkout_ref:
                subprocess.check_call(['git', 'checkout', checkout_ref], cwd=cloned_path)
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repository at {repo_url}. Exiting.")
            exit()
        
        gromet_collection = easy_process_file_system(system_name, str(cloned_path), write_to_file, original_source, dependency_depth)

    return gromet_collection

def process_archive(archive_url: str, write_to_file: bool = False, original_source: bool = False, dependency_depth=0) -> GrometFNModuleCollection:
    """
    Downloads a repository archive in tar format and ingests it into a GrometFNModuleCollection with an optional dependency depth.
    """
    system_name = archive_url.split('/')[-1].replace('.tar.gz', '').replace('.tar', '')

    response = requests.get(archive_url)
   
    with tempfile.TemporaryDirectory() as temp:
        temp_archive_path = Path(temp) / f"{system_name}.tar.gz"
        temp_archive_path.write_bytes(response.content)
        with tarfile.open(temp_archive_path, "r:*") as tar:  
            tar.extractall(path=temp)
        
        extracted_dir_path = Path(temp) / system_name 
        gromet_collection = easy_process_file_system(system_name, str(extracted_dir_path), write_to_file, original_source, dependency_depth)

    return gromet_collection

def main():
    parser = argparse.ArgumentParser(description="Process a Git repository or a tar archive into a GrometFNModuleCollection.")
    parser.add_argument("--mode", choices=['git', 'tar'], required=True, help="The mode of operation: 'git' for Git repositories, 'tar' for tar archives.")
    parser.add_argument("url", help="The URL of the Git repository or tar archive to process.")
    parser.add_argument("--ref", help="The tag, commit, or branch to checkout after cloning (Git mode only).", default=None)
    parser.add_argument("--write_to_file", action="store_true", help="Whether to output Gromet to file.")
    parser.add_argument("--source", action="store_true", help="Toggle whether or not to include the full source code in the Gromet metadata.")
    parser.add_argument("--dependency_depth", type=int, default=0, help="Specify the dependency depth for analysis")

    args = parser.parse_args()
    
    if args.mode == 'git':
        process_git_repo(args.url, args.ref, args.write_to_file, args.source, args.dependency_depth)
    elif args.mode == 'tar':
        process_archive(args.url, args.write_to_file, args.source, args.dependency_depth)
    else:
        print("Invalid mode selected. Please choose either 'git' or 'tar'.")

if __name__ == "__main__":
    main()
