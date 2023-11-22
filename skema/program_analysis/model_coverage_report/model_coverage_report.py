import argparse
import os
import traceback  # Debugs
import requests
import yaml
import json
from enum import Enum
from typing import List, Dict, Tuple, Callable, Any
from zipfile import ZipFile
from io import BytesIO
from tempfile import TemporaryDirectory
from pathlib import Path

from func_timeout import func_timeout, FunctionTimedOut
from tree_sitter import Language, Parser, Tree

from skema.program_analysis.CAST2FN.cast import CAST
from skema.program_analysis.run_ann_cast_pipeline import ann_cast_pipeline
from skema.program_analysis.model_coverage_report.html_builder import HTML_Instance
from skema.program_analysis.multi_file_ingester import process_file_system
from skema.program_analysis.single_file_ingester import process_file
from skema.program_analysis.snippet_ingester import process_snippet
from skema.program_analysis.tree_sitter_parsers.build_parsers import (
    INSTALLED_LANGUAGES_FILEPATH,
    LANGUAGES_YAML_FILEPATH,
)
from skema.program_analysis.python2cast import python_to_cast
from skema.program_analysis.fortran2cast import fortran_to_cast
from skema.program_analysis.matlab2cast import matlab_to_cast
from skema.program_analysis.tree_sitter_parsers.util import extension_to_language
from skema.rest.utils import fn_preprocessor
from skema.utils.fold import del_nulls, dictionary_to_gromet_json

THIS_PATH = Path(__file__).parent.resolve()
MODEL_YAML_PATH = Path(__file__).parent / "models.yaml"
MODEL_YAML = yaml.safe_load(MODEL_YAML_PATH.read_text())

class Status(Enum):
    """Status enum for the status of executing a step in the code2fn pipeline"""

    VALID = "Valid"
    TIMEOUT = "Timeout"
    EXCEPTION = "Exception"

    @staticmethod
    def all_valid(status_list: List) -> bool:
        """Check if all status in a List are Status.VALID"""
        return all([status == Status.VALID for status in status_list])

    @staticmethod
    def get_overall_status(status_list: List) -> str:
        """Return the final pipeline status given a List of status for each step in the pipeline"""
        return (
            Status.TIMEOUT
            if Status.TIMEOUT in status_list
            else Status.EXCEPTION
            if Status.EXCEPTION in status_list
            else Status.VALID
        )


def generate_data_product(
    output_dir: str, model_name: str, file_name: str, data_product_function: Callable, args=(), kwargs=None
) -> Tuple[str, Any]:
    """Wrapper function for generating data products, returns the status of processing."""
    (output, output_path, status) = (None, None, None)

    output_path = Path(output_dir) / "data"/ data_product_function.__name__ / model_name / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    relative_output_path = output_path.relative_to(output_dir)

    # There is a possibility that the processing function fails after changing the working directory.
    # So we should change it back before and after each itteraton.
    os.chdir(THIS_PATH)
    try:
        output = func_timeout(10, data_product_function, args=args, kwargs=kwargs)
        if output == "":
            raise Exception("Data product is empty")
        output_path.write_text(output)
        status = Status.VALID
    except FunctionTimedOut:
        os.chdir(THIS_PATH)
        output_path.write_text("Processing exceeded timeout (10s)")
        status = Status.TIMEOUT
    except (Exception, SystemExit) as e:
        os.chdir(THIS_PATH)
        output_path.write_text(traceback.format_exc())
        status = Status.EXCEPTION
    
    return output, relative_output_path, status


def generate_parse_tree(source: str, language_name: str) -> str:
    """Generator function for Tree-Sitter parse tree"""       
    # Determine the tree-sitter parser we need to use based on file extension
    parser = Parser()
    parser.set_language(Language(INSTALLED_LANGUAGES_FILEPATH, language_name))
    tree = parser.parse(bytes(source, encoding="utf-8"))

    return tree.root_node.sexp()


def generate_cast(source_path: str, language_name: str) -> str:
    """Generator function for CAST"""
    if language_name == "python":
        cast = python_to_cast(source_path, cast_obj=True)
    elif language_name == "fortran":
        cast = fortran_to_cast(source_path, cast_obj=True)
    elif language_name == "matlab":
        cast = matlab_to_cast(source_path, cast_obj=True)

    # Currently, the CAST frontends can either return a single CAST object, or a List of CAST objects.
    if isinstance(cast, List):
        return "\n".join([cast_obj.to_json_str() for cast_obj in cast])
    else:
        return cast.to_json_str()


def generate_gromet(source_path: str) -> str:
    """Generator function for Gromet"""
    gromet_collection = process_file(source_path)
    return dictionary_to_gromet_json(del_nulls(gromet_collection.to_dict()))


def generate_full_gromet(system_name: str, root_path: str, system_filepaths: str):
    """Generator function for Gromet for full system ingest"""
    gromet_collection = process_file_system(system_name, root_path, system_filepaths)
    return dictionary_to_gromet_json(del_nulls(gromet_collection.to_dict()))

def generate_gromet_preprocess_fn(gromet_collection: Dict) -> str:
    """Generator function for Gromet preprocessed fn"""
    gromet_collection = fn_preprocessor(gromet_collection)[0]
    return dictionary_to_gromet_json(del_nulls(gromet_collection))

def generate_gromet_preprocess_logs(gromet_collection: Dict) -> str:
    """Generator function for Gromet preprocessing logs"""
    logs = fn_preprocessor(gromet_collection)[1]
    return "\n".join(logs)

def process_single_model(html: HTML_Instance, output_dir: str, model_name: str):
    """Generate an HTML report for a single model"""
    html.add_model(model_name)

    if model_name in MODEL_YAML:
        model_url = MODEL_YAML[model_name]["zip_archive"]
        response = requests.get(model_url)
    else:
        pass

    zip = ZipFile(BytesIO(response.content))
    with TemporaryDirectory() as temp:
        file_status_list = []
        supported_lines = 0
        total_lines = 0
        for file in zip.filelist:
            source = str(zip.open(file).read(), encoding="utf-8")
            temp_path = Path(temp) / file.filename
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_text(source)
            filename = Path(file.filename).stem

            # Determine the language name by cross referencing the file extension in languages.yaml
            file_extension = Path(file.filename).suffix
            language_name = extension_to_language(file_extension)

            # Step 1: Generate Tree-Sitter parse tree
            # NOTE: This currently produces the parse-tree BEFORE preprocessing. Once we have a generalized preprocessor, we can improve this.
            parse_tree_output, parse_tree_relative_path, parse_tree_status = generate_data_product(
                output_dir, model_name, f"{filename}.txt",generate_parse_tree, args=(source, language_name), kwargs=None
            )

            # Step 2: Generate CAST
            # NOTE: Currently we don't have a system to pass a parse tree to the CAST frontends, so some processing will be repeated.
            cast_output, cast_relative_path, cast_status = generate_data_product(
                output_dir,
                model_name,
                f"{filename}.json",
                generate_cast,
                args=(str(temp_path), language_name),
                kwargs=None,
            )

            # Step 3: Generate Gromet
            # NOTE: The CAST->Gromet function currently only accepts a single CAST object. So we are not currently passing the CAST from the previous step.
            gromet_output, gromet_relative_path, gromet_status = generate_data_product(
                output_dir, model_name, f"{filename}.json", generate_gromet, args=(str(temp_path),), kwargs=None
            )

            # Step 4: Preprocess Gromet
            try: 
                gromet_collection = json.loads(gromet_output)
            except:
                gromet_collection = {}

            gromet_report_output, gromet_report_relative_path, gromet_report_status = generate_data_product(
                output_dir, model_name, f"{filename}.txt", generate_gromet_preprocess_logs, args=(gromet_collection,), kwargs=None 
            )
            gromet_preprocess, gromet_preprocess_relative_path, gromet_preprocess_status = generate_data_product(
                output_dir, model_name, f"{filename}.json", generate_gromet_preprocess_fn, args=(gromet_collection,), kwargs=None 
            )
            
            try:
                gromet_error_count = len(gromet_report_output.splitlines()) # len(gromet_report_output[1].splitlines())
            except:
                gromet_error_count = 0 

            # Check the status of each pipeline step
            final_status = Status.get_overall_status(
                [cast_status, gromet_status]
            )
            file_status_list.append(final_status)

            if final_status == Status.VALID:
                can_ingest = True
                supported_lines += len(source.splitlines())
            else:
                can_ingest = False
            total_lines += len(source.splitlines())

            html.add_file_basic(
                model_name,
                file.filename,
                len(source.splitlines()),
                can_ingest,
                parse_tree_relative_path,
                cast_relative_path,
                gromet_relative_path,
                gromet_error_count,
                gromet_report_relative_path,
                gromet_preprocess_relative_path
            )

        # If all files are valid in a system, attempt to ingest full system into single GrometFNModuleCollection
        if not Status.all_valid(file_status_list):
            html.add_model_header_data(
                model_name, supported_lines, total_lines, False, None
            )
        else:
    
            system_filepaths = Path(temp) / "system_filepaths.txt"
            system_filepaths.write_text(
                "\n".join([file.filename for file in zip.filelist])
            )
            full_gromet, full_gromet_relative_path, full_gromet_status = generate_data_product(
                output_dir,
                model_name,
                f"{model_name}.json",
                generate_full_gromet,
                args=(model_name, str(Path(temp)), str(system_filepaths)),
                kwargs=None,
            )
            
            html.add_model_header_data(
                model_name,
                supported_lines,
                total_lines,
                True,
                full_gromet_relative_path,
            )
        return (supported_lines, total_lines)


def process_all_models(html: HTML_Instance, output_dir: str):
    """Generate an HTML report for all models in models.yaml"""
    model_line_coverage = {}
    for model_name in MODEL_YAML:
        supported, total = process_single_model(html, output_dir, model_name)
        model_line_coverage[model_name] = (supported, total)

    return model_line_coverage


def main():

    parser = argparse.ArgumentParser(description="Process models.")
    parser.add_argument("output_dir", help="Path to the output directory")
    subparsers = parser.add_subparsers(dest="mode")

    # Subparser for the "all" mode
    all_parser = subparsers.add_parser("all")

    # Subparser for the "single" mode
    single_parser = subparsers.add_parser("single")
    single_parser.add_argument("model_name", help="Name of the model to be processed")

    args = parser.parse_args()

    # output_dir has to be resolved ahead of time due to how the cwd is changed in the Gromet pipeline
    output_dir = str(Path(args.output_dir).resolve()) 

    html = HTML_Instance()
    model_line_coverage = {}
    if args.mode == "all":
        model_line_coverage = process_all_models(html, output_dir)
    elif args.mode == "single":
        supported, total = process_single_model(html, output_dir, args.model_name)
        model_line_coverage[args.model_name] = (supported, total)

    # DataTables have to be initialized after all models are generated
    html.add_data_table_script()
    
    output_path = Path(output_dir) / "report.html"
    output_path.write_text(html.soup.prettify())

    json_output_path = Path(output_dir) / "line_coverage.json"
    json_output_path.write_text(json.dumps(model_line_coverage))


if __name__ == "__main__":
    main()
