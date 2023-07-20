# build the tree-sitter MatLab shared object file

from tree_sitter import Language

def main():
    Language.build_library("build/ts-matlab.so",["."])


if __name__ == "__main__":
    main()
