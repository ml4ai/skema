## multi_file_ingester
### Command line arguments
 - **sysname (str)** - The name of the system being ingested
 - **path (str)** - The path to the root of the system
 - **files (str)** - The path to system_filepaths.txt
### system_filepaths.txt
Processing a multi-file system requires a system_filepaths.txt file describing the structure of the system. Each line represents the path to one file in the system relative to the root directory. For example the system_filepaths.txt file for chime_penn_full would be:
```
cli.py
constants.py
model/parameters.py
model/sir.py
model/validators/base.py
model/validators/validators.py
```
### Running as script
```bash
python multi_file_ingester.py --sysname "CHIME" --path /path/to/root --files /path/to/system_filepaths.txt
```
### Running as library
```python
from skema.program_analysis.multi_file_ingester import process_file_system
gromet_collection = process_file_system("CHIME", "data/chime/", "data/chime/system_filepaths.txt", write_to_file=True)
```

## single_file_ingester
### Command line arguments
 - **path (str)** - The relative or absolute path of the file to process"
### Running as script
```bash
python single_file_ingester.py data/TIEGCM/cpktkm.F
```
### Running as library
```python
from skema.program_analysis.single_file_ingester import process_file
gromet_collection = process_file("cpktkm.F", write_to_file=True)
```

## snippet_file_ingester
### Command line arguments
 - **snippet(str)** - The snippet of Python/Fortran code to process"
 - **extension(str)** - A file extension representing the language of the code snippet(.f95, .f, .py)"
### Running as script
```bash
python snippet_file_ingester.py "x=2" ".py"
```
### Running as library
```python
from skema.program_analysis.snippet_file_ingester import process_snippet
gromet_collection = process_snippet("x=2", ".py", write_to_file=True)