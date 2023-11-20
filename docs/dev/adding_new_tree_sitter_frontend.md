## Language support
If the language you wish to use does not already have a [Tree-sitter parser](https://tree-sitter.github.io/tree-sitter/#parsers), you can create it with a grammar for that language.
## Building the Tree-sitter parser
**Requirements**:
* A GitHub repository with a grammar file named `grammar.js` for the language you wish to support. 
* Tree-sitter also support writing your own grammar file from scratch with the steps [shown here.](https://tree-sitter.github.io/tree-sitter/creating-parsers)

**Steps**:
1. In directory `skema/program_analysis/tree_sitter_parsers/` do the following:
2. Add new entry to `languages.yaml`
```yaml
matlab:
  tree_sitter_name: tree-sitter-matlab
  clone_url: https://github.com/acristoffers/tree-sitter-matlab.git
  supports_comment_extraction: True
  supports_fn_extraction: True
  extensions:
    - .m
```
3. Run  `build_parsers.py`. Adding an entry to `languages.yaml` will automatically create a new command line argument for `build_parsers.py`.
```bash
python  build_languages.py --matlab
```
If successful, a build directory will have been created with a language object file `installed_languages.so` 

## Using the tree-sitter parser
**Requirements:**
* Tree-sitter language object file built using above steps

**Steps:**
1. Import the path to the tree-sitter library. 
```python
from skema.program_analysis.CAST.tree_sitter_parsers.build_parsers import INSTALLED_LANGUAGES_FILEPATH
```
2. Create the Language object. This is used for parsing or running queries.
```python
language_object = Language(INSTALLED_LANGUAGES_FILEPATH, "matlab")
```
3. Parse the source code using the language object created above. Note that the source code needs to be a bytes object rather than a string.
```python
parser = Parser()
parser.set_language(language_object)
tree = parser.parse(bytes(source, "utf8"))
```

## Notes on walking tree-sitter Tree
* Running parse will create a Tree of Node objects with the root node stored at tree.root_node. 
* Node objects only contain the fields `type`, `children`, `start_point`, `end_point`. To get the actual string identifier of a node, you need to infer it from the source code and the source reference information. The following is the implementation that the Fortran frontend uses. 
```python
def get_identifier(self, node: Node, source: str) -> str:
        """Given a node, return the identifier it represents. ie. The code between node.start_point and node.end_point"""
        line_num = 0
        column_num = 0
        in_identifier = False
        identifier = ""
        for i, char in enumerate(source):
            if line_num == node.start_point[0] and column_num == node.start_point[1]:
                in_identifier = True
            elif line_num == node.end_point[0] and column_num == node.end_point[1]:
                break

            if char == "\n":
                line_num += 1
                column_num = 0
            else:
                column_num += 1

            if in_identifier:
                identifier += char

        return identifier
```