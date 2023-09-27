import json
import requests
import zipfile
import io
from pathlib import Path
from tempfile import TemporaryFile, TemporaryDirectory

from skema.program_analysis.multi_file_ingester import process_file_system
from skema.gromet.fn import GrometFNModuleCollection

BUCKY_ZIP_URL = "https://artifacts.askem.lum.ai/askem/data/models/zip-archives/Bucky.zip"

def test_code2fn():
    """This is simply a smokescreen test to see if the PA pipeline runs to
    completion without crashing. It does not test the actual outputs.

    This is because the output JSON contains randomly generated UUIDs, making
    deterministic testing difficult. In the future, it would be good to add a
    flag or postprocessing function to be able to compare two GroMEts modulo
    their random components."""

    response = requests.get(BUCKY_ZIP_URL)
    zip = zipfile.ZipFile(io.BytesIO(response.content))

    with TemporaryDirectory() as temp:
        system_filepaths_path = Path(temp) / "system_filepaths.txt"
        system_filepaths_path.write_text("\n".join([file.filename for file in zip.filelist]))

        for file in zip.filelist:
            file_path = Path(temp) / file.filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(str(zip.open(file, "r").read(), encoding="utf-8"))

        module_collection: GrometFNModuleCollection = process_file_system(
            "chime_penn",
            str(temp),
            str(system_filepaths_path),
        )

    # If we've made it this far the Gromet pipeline has run without crashing
    assert True