# Clone and build the treesitter matlab grammar

import os.path
import subprocess
import shutil
from tree_sitter import Language

MATLAB_CLONE_URL = 'https://github.com/acristoffers/tree-sitter-matlab.git'
MATLAB_TEST_URL = 'https://github.com/mathworks/MATLAB-Language-grammar.git' 

SHARED_OBJECT_DIR = 'build'
LANGUAGE_LIBRARY_REL_PATH = os.path.join(SHARED_OBJECT_DIR, "ts-matlab.so")

# Determine the caller path
CALLER_PATH = os.getcwd()

# Determine absolute path of this file
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

# given a git repo URL return the source directory name
def git_dir_from_url(git_url):
    return git_url.split('/')[-1].split('.git')[0]

# whack a directory
def clean(target):
    ret = subprocess.run(['rm', '-rf', target])
    if (ret):
        print('Cleaned ' + target)
    else:
        print('Could not clean ' + target)
    return ret

# clone a git repo
def clone(git_url):
    # Clone the repo from github
    print('Cloning ' + git_url)
    ret = subprocess.run(['git', 'clone', git_url])

    if(ret):
        print('Clone succeeded')
    else:
        print('Clone failed')
    return ret

def build_matlab_grammar():
    Language.build_library(
        output_path=LANGUAGE_LIBRARY_REL_PATH,
        repo_paths=['tree-sitter-matlab']
    )

# Test the grammar using the test corpus 
def run_matlab_test_corpus():
    pass

def main():
    # Move to project directory
    os.chdir(PROJECT_PATH)

    # Clean the target directories
    clean(SHARED_OBJECT_DIR)
    clean(git_dir_from_url(MATLAB_CLONE_URL))
    clean(git_dir_from_url(MATLAB_TEST_URL))

    # create the build directory
    ret = subprocess.run(['mkdir', SHARED_OBJECT_DIR])

    # Clone the tree-sitter matlab grammar and test corpus repos
    clone(MATLAB_CLONE_URL)
    clone(MATLAB_TEST_URL)

    # Build the matlab shared object file
    build_matlab_grammar()

    run_matlab_test_corpus()


    # Return to caller directory
    os.chdir(CALLER_PATH)

if __name__ == "__main__":
    main()
