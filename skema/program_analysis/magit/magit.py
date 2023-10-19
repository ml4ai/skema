import argparse
import os
import traceback # Debugs
import requests
import yaml
from zipfile import ZipFile
from io import BytesIO
from tempfile import TemporaryDirectory
from pathlib import Path

from tree_sitter import Language, Parser, Tree

from skema.program_analysis.magit.html_builder import HTML_Instance
from skema.program_analysis.multi_file_ingester import process_file_system
from skema.program_analysis.single_file_ingester import process_file
from skema.program_analysis.snippet_ingester import process_snippet
from skema.program_analysis.tree_sitter_parsers.build_parsers import INSTALLED_LANGUAGES_FILEPATH, LANGUAGES_YAML_FILEPATH
from skema.program_analysis.python2cast import python_to_cast

THIS_PATH = Path(__file__).resolve().parent
MODEL_YAML_PATH = Path(__file__).parent / "models.yaml"
MODEL_YAML = yaml.safe_load(MODEL_YAML_PATH.read_text())

def generate_data_products(output_dir: str, source: str, file_name: str):
    """Generate the data produce directory for the html output"""
    file_path = Path(file_name)
    language_name = ""
    
    parse_tree_path = Path(output_dir) / "data" / "tree-sitter" / f"{file_path.stem}.txt"
    parse_tree_path.parent.mkdir(parents=True, exist_ok=True)
    cast_path = Path(output_dir) / "data" / "cast" / f"{file_path.stem}.json"
    cast_path.parent.mkdir(parents=True, exist_ok=True)
    gromet_path = Path(output_dir) / "data" / "gromet" / f"{file_path.stem}.json"
    gromet_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine the tree-sitter parser we need to use based on file extension
    file_extension = Path(file_name).suffix
    yaml_obj = yaml.safe_load(LANGUAGES_YAML_FILEPATH.read_text())
    for language, language_dict in yaml_obj.items():
        if file_extension in language_dict["extensions"]:
            language_name = language
            parser = Parser()
            parser.set_language(Language(INSTALLED_LANGUAGES_FILEPATH, language))
            tree = parser.parse(bytes(source, encoding="utf-8"))
            parse_tree_path.write_text(tree.root_node.sexp())

    # Relative to report/
    return (str(parse_tree_path.relative_to("report/")), str(cast_path.relative_to("report/")), str(gromet_path.relative_to("report/")))
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
            try:
                process_file(temp_path)
                can_ingest = True
                supported_lines += len(source.splitlines())
            except (SystemExit, Exception) as e:
                can_ingest = False
            total_lines += len(source.splitlines())

            # The file ingestors modify the cwd during processing. If there is a failure, then the cwd may not be changed back.
            # This can result in the cwd begin a temp file that is deleted.
            # To account for this, we will set the cwd to the current directory
            # os.chdir(Path("."))
            os.chdir(THIS_PATH)
            
            # Generate the data products for this report
            output_paths = generate_data_products(output_dir, source, file.filename)
            
            html.add_file_basic(model_name, file.filename, len(source.splitlines()), can_ingest, output_paths[0], output_paths[1], output_paths[2])
        
        # Generate top level model header data
        system_filepaths = Path(temp) / "system_filepaths.txt"
        system_filepaths.write_text("\n".join([file.filename for file in zip.filelist]))
        try:
            process_file_system(model_name, Path(temp), system_filepaths)
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
    