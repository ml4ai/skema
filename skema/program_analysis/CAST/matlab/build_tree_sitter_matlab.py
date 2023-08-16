# Clone and build the treesitter matlab grammar

import os.path
import subprocess
import shutil
from tree_sitter import Language

MATLAB_CLONE_URL = 'https://github.com/acristoffers/tree-sitter-matlab.git'
MATLAB_TEST_CORPUS_URL = 'https://github.com/mathworks/MATLAB-Language-grammar.git' 

LANGUAGE_LIBRARY_REL_PATH = os.path.join("build", "ts-matlab.so")

# Determine the caller path
CALLER_PATH = os.getcwd()

# Determine absolute path of this file
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

# given a git repo URL return the source directory name
def target_dir(git_url):
    # TODO try block
    return git_url.split('/')[-1].split('.git')[0]

# clone a git repo
def clone(git_url):

    # Remove the repo from local filesystem if it exists
    target = target_dir(git_url)
    print('Cleaning ' + target)
    ret = subprocess.run(['rm', '-rf', target])

    # Clone the repo from github
    print('Cloning ' + git_url)
    ret = subprocess.run(['git', 'clone', git_url])

    if(ret):
        print('Clone succeeded')
    else:
        print('Clone failed')
    return ret


def build_matlab_grammar():
    Language.build_library(output_path='build/ts-matlab.so', repo_paths=['tree-sitter-matlab'])


def run_matlab_test_corpus():
    pass

def main():
    # Move to project directory
    os.chdir(PROJECT_PATH)

    # Clone the tree-sitter matlab grammar and test corpus repos
    clone(MATLAB_CLONE_URL)
    clone(MATLAB_TEST_CORPUS_URL)

    # Build the matlab shared object file
    build_matlab_grammar()

    run_matlab_test_corpus()


    # Return to caller directory
    os.chdir(CALLER_PATH)

if __name__ == "__main__":
    main()
