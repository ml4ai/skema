import json
from pathlib import Path
from skema.program_analysis.multi_file_ingester import process_file_system
from skema.gromet.fn import GrometFNModuleCollection

def test_code2fn():
    """This is simply a smokescreen test to see if the PA pipeline runs to
    completion without crashing. It does not test the actual outputs.

    This is because the output JSON contains randomly generated UUIDs, making
    deterministic testing difficult. In the future, it would be good to add a
    flag or postprocessing function to be able to compare two GroMEts modulo
    their random components."""

    data_dir = Path(__file__).parents[3] / "data"
    module_collection: GrometFNModuleCollection = process_file_system(
        "chime_penn",
        str(data_dir / "epidemiology/Bucky/code/bucky_v2"),
        str(data_dir / "epidemiology/Bucky/code/system_filepaths.txt"),
    )
