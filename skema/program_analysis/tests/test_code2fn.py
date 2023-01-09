import json
from skema.program_analysis.multi_file_ingester import process_file_system
from skema.gromet.fn import GrometFNModuleCollection


def test_code2fn():
    module_collection: GrometFNModuleCollection = process_file_system(
        "chime_penn",
        "../../data/epidemiology/CHIME/CHIME_penn_full_model/code/penn_chime",
        "../../data/epidemiology/CHIME/CHIME_penn_full_model/code/system_filepaths.txt",
    )

    with open(
        "../../data/epidemiology/CHIME/CHIME_penn_full_model/gromet/chime_penn--Gromet-FN-auto.json"
    ) as f:
        test_data = json.load(f)

    assert test_data == module_collection.to_dict()
