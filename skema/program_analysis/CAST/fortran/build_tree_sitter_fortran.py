# Simple script to clone and build tree-sitter-fortran respository

import subprocess
import os.path
from tree_sitter import Language, Parser

# TODO: Eventually, we can look into locking this to a specific version
FORTRAN_CLONE_URL = "https://github.com/stadelmanma/tree-sitter-fortran.git"
LANGUAGE_LIBRARY_REL_PATH = os.path.join("build", "ts-fortran.so")

# We are using __file__ to guarantee that the library is created relative to the CAST/fortran directory no matter where its called from
root_path = os.path.dirname(os.path.abspath(__file__))


def main():
    # Set working directory to CAST/fortran
    wd = os.getcwd()
    os.chdir(root_path)

    subprocess.run(["git", "clone", FORTRAN_CLONE_URL])

    Language.build_library(
        # Store the library in the `build` directory
        LANGUAGE_LIBRARY_REL_PATH,
        # Include one or more languages
        ["tree-sitter-fortran"],
    )

    # Reset working directory
    os.chdir(wd)


if __name__ == "__main__":
    main()
