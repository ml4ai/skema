import requests
import zipfile
import io
from tempfile import TemporaryFile, TemporaryDirectory
from pathlib import Path

from tree_sitter import Language, Parser

from skema.program_analysis.tree_sitter_parsers.build_parsers import (
    INSTALLED_LANGUAGES_FILEPATH,
)
from skema.program_analysis.CAST.fortran.preprocessor.preprocess import preprocess

TIE_GCM_URL = (
    "https://artifacts.askem.lum.ai/askem/data/models/zip-archives/TIE-GCM.zip"
)


def validate_parse_tree(source: str) -> bool:
    """Parse source with tree-sitter and check if an error is returned."""
    language = Language(INSTALLED_LANGUAGES_FILEPATH, "fortran")
    parser = Parser()
    parser.set_language(language)
    tree = parser.parse(bytes(source, encoding="utf-8"))
    return "ERROR" not in tree.root_node.sexp()


def test_parse_tiegcm():
    response = requests.get(TIE_GCM_URL)
    zip = zipfile.ZipFile(io.BytesIO(response.content))

    # We will run two sets of tests:
    # 1. Parse with tree-sitter as-is
    # 2. Parse with tree-sitter after preprocessing
    for file in zip.filelist:
        if file.filename.endswith(".py"):
            continue
        source = str(zip.open(file).read(), encoding="utf-8")
        with TemporaryDirectory() as temp:
            temp_path = Path(temp) / file.filename
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_text(source)
            try:
                preprocess_source = preprocess(temp_path)
            except:
                print(file.filename)
                print("PREPROCESS FAILURE")
                continue

        parsable = validate_parse_tree(source)
        preprocess_parsable = validate_parse_tree(preprocess_source)

        print(file.filename)
        print(f"original: {parsable}")
        print(f"preprocess: {preprocess_parsable}")

        assert parsable or preprocess_parsable