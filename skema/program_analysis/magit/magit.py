import argparse
import os
import traceback # Debugs
import requests
import yaml
from zipfile import ZipFile
from io import BytesIO
from tempfile import TemporaryDirectory
from pathlib import Path

from skema.program_analysis.magit.html_builder import HTML_Instance
from skema.program_analysis.multi_file_ingester import process_file_system
from skema.program_analysis.single_file_ingester import process_file
from skema.program_analysis.snippet_ingester import process_snippet

THIS_PATH = Path(__file__).resolve().parent
MODEL_YAML_PATH = Path(__file__).parent / "models.yaml"
MODEL_YAML = yaml.safe_load(MODEL_YAML_PATH.read_text())

def process_single_model(html: HTML_Instance, model_name: str):
    html.add_model(model_name)

    if model_name in MODEL_YAML:
        model_url = MODEL_YAML[model_name]["zip_archive"]
        response = requests.get(model_url)
    zip = ZipFile(BytesIO(response.content))
    with TemporaryDirectory() as temp:
        system_filepaths = Path(temp) / "system_filepaths.txt"
        system_filepaths.write_text("\n".join([file.filename for file in zip.filelist]))
        for file in zip.filelist:
            source = str(zip.open(file).read(), encoding="utf-8")
            temp_path = Path(temp) / file.filename
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_text(source)
            try:
                process_file(temp_path)
                can_ingest = True
            except (SystemExit, Exception) as e:
                can_ingest = False

            # The file ingestors modify the cwd during processing. If there is a failure, then the cwd may not be changed back.
            # This can result in the cwd begin a temp file that is deleted.
            # To account for this, we will set the cwd to the current directory
            # os.chdir(Path("."))
            os.chdir(THIS_PATH)
            html.add_file_basic(model_name, file.filename, len(source), can_ingest)
    
    

def process_all_models(html: HTML_Instance):
    for model_name in MODEL_YAML:
        print(f"PROCESSING {model_name}")
        process_single_model(html, model_name)
        print(f"Done {model_name}")
    print("FINSIHED Processing modesl")

"""
html = HTML_Instance()

for model, model_dict in MODEL_YAML.items():
    html.add_model(model)

    model_url = model_dict["zip_archive"]
    response = requests.get(model_url)
    zip = ZipFile(BytesIO(response.content))
    with TemporaryDirectory() as temp:
        system_filepaths = Path(temp) / "system_filepaths.txt"
        system_filepaths.write_text("\n".join([file.filename for file in zip.filelist]))
        for file in zip.filelist:
            source = str(zip.open(file).read(), encoding="utf-8")
            temp_path = Path(temp) / file.filename
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_text(source)
            
            try:
                process_file(temp_path)
                can_ingest = True
            except:
                can_ingest = False
            
            html.add_file_basic(model, file.filename, len(source), can_ingest)

html.write_html()
"""
if __name__  == "__main__":
    parser = argparse.ArgumentParser(description="Process models.")
    subparsers = parser.add_subparsers(dest="mode")
    
    # Subparser for the "all" mode
    all_parser = subparsers.add_parser("all")

     # Subparser for the "single" mode
    single_parser = subparsers.add_parser("single")
    single_parser.add_argument("model_name", help="Name of the model to be processed")

    args = parser.parse_args()
    
    html = HTML_Instance()
    if args.mode == "all":
        process_all_models(html)
    elif args.mode == "single":
        process_single_model(html, args.model_name)
    html.write_html()