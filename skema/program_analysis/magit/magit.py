import argparse
import os
import traceback # Debugs
import requests
import yaml
from typing import List, Tuple, Callable
from zipfile import ZipFile
from io import BytesIO
from tempfile import TemporaryDirectory
from pathlib import Path

from func_timeout import func_timeout, FunctionTimedOut
from tree_sitter import Language, Parser, Tree

from skema.program_analysis.CAST2FN.cast import CAST
from skema.program_analysis.run_ann_cast_pipeline import ann_cast_pipeline
from skema.program_analysis.magit.html_builder import HTML_Instance
from skema.program_analysis.multi_file_ingester import process_file_system
from skema.program_analysis.single_file_ingester import process_file
from skema.program_analysis.snippet_ingester import process_snippet
from skema.program_analysis.tree_sitter_parsers.build_parsers import INSTALLED_LANGUAGES_FILEPATH, LANGUAGES_YAML_FILEPATH
from skema.program_analysis.python2cast import python_to_cast
from skema.program_analysis.fortran2cast import fortran_to_cast
from skema.program_analysis.matlab2cast import matlab_to_cast
from skema.utils.fold import del_nulls, dictionary_to_gromet_json

THIS_PATH = Path(__file__).resolve().parent
MODEL_YAML_PATH = Path(__file__).parent / "models.yaml"
MODEL_YAML = yaml.safe_load(MODEL_YAML_PATH.read_text())

def generate_data_product(output_path: Path, data_product_function: Callable, args=(), kwargs=None) -> str:
    """Wrapper function for generating data products, returns the status of processing."""
    try:
        output = func_timeout(60, data_product_function, args=args, kwargs=kwargs)
        output_path.write_text(output)
        return "Valid"
    except FunctionTimedOut:
        output_path.write_text("Processing exceeded timeout (60s)")
        return "Timeout"
    except (Exception, SystemExit) as e:
        traceback.print_exc()
        output_path.write_text(traceback.format_exc())
        return "Exception"

def generate_parse_tree(source: str, language_name: str) -> str:
    """Generator function for Tree-Sitter parse tree"""
    
    # Determine the tree-sitter parser we need to use based on file extension
    """file_extension = source_path.suffix
    yaml_obj = yaml.safe_load(LANGUAGES_YAML_FILEPATH.read_text())
    for language, language_dict in yaml_obj.items():
        if file_extension in language_dict["extensions"]:
            language_name = language
    """
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

def process_single_model(html: HTML_Instance, output_dir: str, model_name: str):
    html.add_model(model_name)

    if model_name in MODEL_YAML:
        model_url = MODEL_YAML[model_name]["zip_archive"]
        response = requests.get(model_url)
        
    zip = ZipFile(BytesIO(response.content))
    with TemporaryDirectory() as temp:
        # Ingest each file individually
        supported_lines = 0
        total_lines = 0
        for file in zip.filelist:
            source = str(zip.open(file).read(), encoding="utf-8")
            temp_path = Path(temp) / file.filename
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_text(source)
            filename = Path(file.filename).stem
            
            # Determine the language name by cross referencing the file extension in languages.yaml
            language_name = ""
            file_extension = Path(file.filename).suffix
            yaml_obj = yaml.safe_load(LANGUAGES_YAML_FILEPATH.read_text())
            for language, language_dict in yaml_obj.items():
                if file_extension in language_dict["extensions"]:
                    language_name = language
            
            # Step 1: Generate Tree-Sitter parse tree
            # NOTE: This currently produces the parse-tree BEFORE preprocessing. Once we have a generalized preprocessor, we can improve this.
            parse_tree_path = Path(output_dir) / "data" / "tree-sitter" / model_name /  f"{filename}.txt"
            parse_tree_path.parent.mkdir(parents=True, exist_ok=True)
            parse_tree_relative_path = str(parse_tree_path.relative_to("report/"))
            parse_tree_status = generate_data_product(parse_tree_path, generate_parse_tree, (source, language_name), None)

            # Step 2: Generate CAST
            # NOTE: Currently we don't have a system to pass a parse tree to the CAST frontends, so some processing will be repeated.
            cast_path = Path(output_dir) / "data" / "cast" / model_name / f"{filename}.json" 
            cast_path.parent.mkdir(parents=True, exist_ok=True)
            cast_relative_path =str(cast_path.relative_to("report/"))
            cast_status = generate_data_product(cast_path, generate_cast, args=(str(temp_path), language_name), kwargs=None)
            
            # Step 3: Generate Gromet
            # NOTE: The CAST->Gromet function currently only accepts a single CAST object. So we are not currently passing the CAST from the previous step.
            gromet_path = Path(output_dir) / "data" / "gromet" / model_name / f"{filename}.json" 
            gromet_path.parent.mkdir(parents=True, exist_ok=True)
            gromet_relative_path = str(gromet_path.relative_to("report/"))
            gromet_status = generate_data_product(gromet_path, generate_gromet, args=(str(temp_path), ), kwargs=None)
            
            # Check the status of each pipeline step
            status_list = [parse_tree_status, cast_status, gromet_status]
            final_status = "Timeout" if "Timeout" in status_list else "Exception" if "Exception" in status_list else "Valid"
            if final_status == "Valid":
                can_ingest = True
                supported_lines += len(source.splitlines())
            else:
                can_ingest = False
            total_lines += len(source.splitlines())


            # The file ingestors modify the cwd during processing. If there is a failure, then the cwd may not be changed back.
            # This can result in the cwd begin a temp file that is deleted.
            # To account for this, we will set the cwd to the current directory
            # os.chdir(Path("."))
            os.chdir(THIS_PATH)
            
            html.add_file_basic(model_name, file.filename, len(source.splitlines()), can_ingest, parse_tree_relative_path, cast_relative_path, gromet_relative_path)
        
        # Attempt to ingest full system into single GrometFNModuleCollection
        system_filepaths = Path(temp) / "system_filepaths.txt"
        system_filepaths.write_text("\n".join([file.filename for file in zip.filelist]))
        try:
            process_file_system(model_name, str(Path(temp)), str(system_filepaths))
            can_ingest_full = True
        except (SystemExit, Exception) as e:
            can_ingest_full = False

        html.add_model_header_data(model_name, supported_lines, total_lines, can_ingest_full )
    


def process_all_models(html: HTML_Instance, output_dir: str):
    for model_name in MODEL_YAML:
        process_single_model(html, output_dir, model_name)

if __name__  == "__main__":
    parser = argparse.ArgumentParser(description="Process models.")
    parser.add_argument("output_dir", help="Path to the output directory")
    subparsers = parser.add_subparsers(dest="mode")
    
    # Subparser for the "all" mode
    all_parser = subparsers.add_parser("all")

     # Subparser for the "single" mode
    single_parser = subparsers.add_parser("single")
    single_parser.add_argument("model_name", help="Name of the model to be processed")

    args = parser.parse_args()
        
    html = HTML_Instance()
    if args.mode == "all":
        process_all_models(html, args.output_dir)
    elif args.mode == "single":
        process_single_model(html, args.output_dir, args.model_name)
    
    output_path = Path(args.output_dir) / "report.html"
    output_path.write_text(html.soup.prettify())
    