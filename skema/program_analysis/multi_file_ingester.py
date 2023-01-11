import argparse
import glob

import os.path

from skema.gromet.fn import (
    GrometFNModuleCollection,
)

from skema.program_analysis.run_ann_cast_pipeline import ann_cast_pipeline
from skema.program_analysis.python2cast import python_to_cast
from skema.utils.fold import dictionary_to_gromet_json, del_nulls


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
        help="If true, the script write the output to a JSON file"
    )

    options = parser.parse_args()
    return options


def process_file_system(
    system_name, path, files, write_to_file=False
) -> GrometFNModuleCollection:
    root_dir = path.strip()
    file_list = open(files, "r").readlines()

    module_collection = GrometFNModuleCollection(
        schema_version="0.1.5",
        name=system_name,
        modules=[],
        module_index=[],
        executables=[],
    )

    for f in file_list:
        full_file = os.path.join(os.path.normpath(root_dir), f.strip("\n"))

        try:
            cast = python_to_cast(full_file, cast_obj=True)
            generated_gromet = ann_cast_pipeline(
                cast, gromet=True, to_file=False, from_obj=True
            )

            # Then, after we generate the GroMEt we store it in the 'modules' field
            # and store its path in the 'module_index' field
            module_collection.modules.append(generated_gromet)

            # DONE: Change this so that it's the dotted path from the root
            # i.e. like model.view.sir" like it shows up in Python
            source_directory = os.path.basename(
                os.path.normpath(root_dir)
            )  # We just need the last directory of the path, not the complete path
            os_module_path = os.path.join(source_directory, f)
            python_module_path = os_module_path.replace("/", ".").replace(
                ".py", ""
            )
            module_collection.module_index.append(python_module_path)

            # Done: Determine how we know a gromet goes in the 'executable' field
            # We do this by finding all user_defined top level functions in the Gromet
            # and check if the name 'main' is among them
            function_networks = [
                fn.value
                for fn in generated_gromet.attributes
                if fn.type == "FN"
            ]
            defined_functions = [
                fn.b[0].name
                for fn in function_networks
                if fn.b[0].function_type == "FUNCTION"
            ]
            if "main" in defined_functions:
                module_collection.executables.append(python_module_path)

        except ImportError:
            print("FAILURE")

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
