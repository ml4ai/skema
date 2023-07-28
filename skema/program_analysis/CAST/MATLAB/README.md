# Tree-sitter MATLAB support
MATLAB is now supported as a tree-sitter grammar for Linux and MacOs operating systems.
This grammar was sourced externally:
- Repository:  https://github.com/acristoffers/tree-sitter-matlab 
- License (MIT): https://github.com/acristoffers/tree-sitter-matlab/blob/main/LICENSE


## Supporting scripts

### clean_grammar
- Remove all generated files, get the build to a clean start

### build_grammar
- Build the grammar from source
- All source files are included in this repository

### test_grammar
- Run the tree-sitter test corpus on the grammar
- Test corpus is included in this repository


## Testing MATLAB code
- First build and test the grammar
- Then execute the following
- ```tree-sitter parse <your_matlab_file.m>```
- If successful you will see the tree-sitter output stream with no errors reported
