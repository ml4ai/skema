import subprocess
import tempfile
import os
import argparse
from pathlib import Path

from skema.gromet.fn import GrometFNModuleCollection
from skema.program_analysis.easy_multi_file_ingester import easy_process_file_system

def process_git_repo(system_name: str, repo_url: str, checkout_ref=None, write_to_file=False, original_source=False) -> GrometFNModuleCollection:
    """
    Clones a Git repository to a temporary directory and ingests it into a GrometFNModuleCollection.

    Parameters:
    system_name (str): The name of the system to ingest.
    repo_url (str): The URL of the Git repository.
    checkout_ref (str, optional): The tag, commit, or branch to checkout after cloning.
    write_to_file (bool, optional): Whether or not to output Gromet to file.
    original_source (bool, optional): Whether or not to include original source code in output Gromet.
    """
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp:
        try:
            # Execute the git clone command
            subprocess.check_call(['git', 'clone', repo_url], cwd=temp)
            if checkout_ref:
                # Checkout the specified tag, commit, or branch
                subprocess.check_call(['git', 'checkout', checkout_ref],cwd=temp)
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repositroy at {repo_url}. Exiting.")
            exit()
        
        gromet_collection = easy_process_file_system(system_name, temp, write_to_file, original_source)

    return gromet_collection


def main():
    parser = argparse.ArgumentParser(description="Clone a Git repository into a temporary directory and optionally check out a specific ref.")
    parser.add_argument("system_name", help="The name of the system to ingest.")
    parser.add_argument("repo_url", help="The URL of the Git repository to clone.")
    parser.add_argument("--ref", help="The tag, commit, or branch to checkout after cloning.", default=None)
    parser.add_argument(
        "--source", action="store_true", help="Toggle whether or not to include the full source code of the code in the GroMEt metadata"
    )
    args = parser.parse_args()
    
    process_git_repo(args.system_name, args.repo_url, args.ref, True, args.source)

if __name__ == "__main__":
    main()