import argparse
import glob
import sys
import os.path
import yaml
from pathlib import Path
from typing import List

from skema.gromet import GROMET_VERSION
from skema.gromet.fn import (
    GrometFNModuleCollection,
)

from skema.program_analysis.run_ann_cast_pipeline import ann_cast_pipeline
from skema.program_analysis.python2cast import python_to_cast
from skema.program_analysis.fortran2cast import fortran_to_cast
from skema.program_analysis.matlab2cast import matlab_to_cast
from skema.utils.fold import dictionary_to_gromet_json, del_nulls
from skema.program_analysis.tree_sitter_parsers.build_parsers import LANGUAGES_YAML_FILEPATH
from skema.program_analysis.module_locate import extract_imports

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sysname", type=str, help="The name of the system we're ingesting"
    )
    parser.add_argument(
        "--path", type=str, help="The path of source directory"
    )
    parser.add_argument(
        "--files",
        type=str,
        help="The path to a file containing a list of files to ingest",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="If true, the script write the output to a JSON file",
    )

    options = parser.parse_args()
    return options


def process_file_system(
    system_name, path, files, write_to_file=False, original_source=False, dependency_depth=0
) -> GrometFNModuleCollection:
    root_dir = path.strip()
    file_list = open(files, "r").readlines()

    module_collection = GrometFNModuleCollection(
        schema_version=GROMET_VERSION,
        name=system_name,
        modules=[],
        module_index=[],
        module_dependencies=[],
        executables=[],
    )

    language_yaml_obj = yaml.safe_load(LANGUAGES_YAML_FILEPATH.read_text())
    cur_dir = os.getcwd()
    for f in file_list:
        full_file = os.path.join(os.path.normpath(root_dir), f.strip("\n"))
        full_file_obj = Path(full_file)

        try:
            # To maintain backwards compatibility for the process_file_system function, for now we will determine the language by file extension
            if full_file_obj.suffix in language_yaml_obj["python"]["extensions"]:
                cast = python_to_cast(full_file, cast_obj=True)
            elif full_file_obj.suffix in language_yaml_obj["matlab"]["extensions"]:
                cast = matlab_to_cast(full_file, cast_obj=True)
            elif full_file_obj.suffix in language_yaml_obj["fortran"]["extensions"]:
                cast = fortran_to_cast(full_file, cast_obj=True)
            else:
                print(f"File extension not supported for {full_file}")

            # The Fortran CAST inteface (CAST/fortran) can produce multiple CAST modules.
            # However, the Python interface (python2cast) will only return a single module.
            # This workaround will normalize a single CAST module into a list for consistent processing.
            if isinstance(cast, List):
                cast_list = cast
            else:
                cast_list = [cast]

            for cast_module in cast_list:
                os.chdir(os.path.join(os.getcwd(), path))
                generated_gromet = ann_cast_pipeline(
                    cast_module, gromet=True, to_file=False, from_obj=True
                )
                os.chdir(cur_dir)

                # NOTE: July '23 Hackathon addition
                # If this flag is set to true, then we read the entire source file into a string, and store it in the 
                if original_source:
                    source_metadata = generated_gromet.metadata_collection[1]
                    # Open the original source code file, read the lines into a list
                    # and then convert back into a string representing the full file    
                    file_text = "".join(open(full_file).readlines())
                    source_metadata[0].files[0].source_string = file_text
                    

                # Then, after we generate the GroMEt we store it in the 'modules' field
                # and store its path in the 'module_index' field
                module_collection.modules.append(generated_gromet)

                # DONE: Change this so that it's the dotted path from the root
                # i.e. like model.view.sir" like it shows up in Python
                source_directory = os.path.basename(
                    os.path.normpath(root_dir)
                )  # We just need the last directory of the path, not the complete path
                os_module_path = os.path.join(source_directory, f)
                       
                # Normalize the path across os and then convert to module dot notation
                python_module_path = ".".join(os.path.normpath(os_module_path).split(os.path.sep))
                python_module_path = ".".join(python_module_path.split(".")[0:-1])
                
                module_collection.module_index.append(python_module_path)

                # TODO: Check for duplicate modules across files
                # TODO: Remove submodule if higher level module is included
                module_collection.module_dependencies.extend(extract_imports(full_file_obj.read_text()))
               
                # Done: Determine how we know a gromet goes in the 'executable' field
                # We do this by finding all user_defined top level functions in the Gromet
                # and check if the name 'main' is among them
                function_networks = [
                    fn
                    for fn in generated_gromet.fn_array
                ]
                defined_functions = [
                    fn.b[0].name
                    for fn in function_networks
                    if fn.b[0].function_type == "FUNCTION"
                ]
                if "main" in defined_functions:
                    module_collection.executables.append(
                        len(module_collection.module_index)
                    )
            

        except (Exception,SystemExit) as e:
            os.chdir(cur_dir)
            print(e)

    def clean_dependencies(dependencies, system_name):
        # Step 1: Remove duplicates and perform initial filtering in one step.
        # This uses a dictionary to preserve insertion order (Python 3.7+ guaranteed order).
        cleaned = {
            dep.name: dep for dep in dependencies
            if not dep.name.startswith(".") and dep.name != system_name
        }.values()

        # Step 2: Sort by the number of dots in the name.
        sorted_deps = sorted(cleaned, key=lambda dep: dep.name.count('.'))

        # Step 3: Remove submodules of other modules.
        # This step keeps an entry if no other entry is its "parent" module.
        final_deps = [
            dep for i, dep in enumerate(sorted_deps)
            if not any(dep.name.startswith(other.name + ".") for other in sorted_deps[:i])
        ]

        return final_deps
 
    module_collection.module_dependencies = clean_dependencies(module_collection.module_dependencies, system_name)
    
    # NOTE: These cannot be imported at the top-level due to circular dependancies
    from skema.program_analysis.single_file_ingester import process_file
    from skema.program_analysis.easy_multi_file_ingester import easy_process_file_system
    from skema.program_analysis.url_ingester import process_git_repo, process_archive
    
    if dependency_depth > 0:
        to_add = []
        for index, dependency in enumerate(module_collection.module_dependencies):

            if dependency.source_reference.type == "Local":
                if Path(dependency.source_reference.value).is_dir():
                    dependency_gromet = easy_process_file_system(dependency.name, dependency.source_reference.value, False, False, dependency_depth=dependency_depth-1)
                else:
                    dependency_gromet = process_file(dependency.source_reference.value, False, False, dependency_depth=dependency_depth-1)
            elif dependency.source_reference.type == "Url":
                dependency_gromet = process_archive(dependency.source_reference.value, False, False, dependency_depth=dependency_depth-1)
            elif dependency.source_reference.type == "Repository":
                dependency_gromet = process_git_repo(dependency.source_reference.value, None, False, False, dependency_depth=dependency_depth-1)
            else:
                continue
            
            # Flatten dependency gromet onto parent Gromet
            for index in range(len(dependency_gromet.modules)):
                dependency_gromet.modules[index].is_depenency = True
            module_collection.modules.extend(dependency_gromet.modules)
            module_collection.module_index.extend([f"{element} (dependency)" for element in dependency_gromet.module_index])
            to_add.extend(dependency_gromet.module_dependencies)
        
        module_collection.module_dependencies.extend(to_add)
    
    if write_to_file:
        with open(f"{system_name}--Gromet-FN-auto.json", "w") as f:
            gromet_collection_dict = module_collection.to_dict()
            f.write(
                dictionary_to_gromet_json(del_nulls(gromet_collection_dict))
            )


    return module_collection


if __name__ == "__main__":
    args = get_args()

    system_name = args.sysname
    path = args.path
    files = args.files

    print(f"Ingesting system: {system_name}")
    print(f"With root directory as specified in: {path}")
    print(f"Ingesting the files as specified in: {files}")

    process_file_system(system_name, path, files, args.write)
