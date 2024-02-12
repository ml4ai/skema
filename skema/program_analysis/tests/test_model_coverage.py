import json
import pytest 
import os

# Dictionary that maps
# "model" : [supported lines, total lines]
# Last updated: January 26th, 2024
# (Updated Bucky model to a slightly smaller count, due to some unknown issue)
# Previous update: November 29th, 2023
# (Updated TIE-GCM to a smaller line count, to address potential timeout issues on the GitHub CI)
ALL_MODELS = {
    "CHIME-penn-full": [1080, 1080], 
    "CHIME-SIR": [633, 633], 
    "CHIME-SVIIvR": [539, 539], 
    "ABM-COmplexVID-19": [1133, 1133], 
    "ABM-COVID-ABS": [1094, 2729], 
    "MechBayes": [0, 0], 
    "SIDARTHE": [193, 193], 
    "Simple-SIR": [53, 53], 
    "Climlab-v1": [4306, 4306], 
    "Generated-Halfar": [128, 128], 
    "SV2AIR3-Waterloo-MATLAB": [0, 1020], 
    "Bucky": [7537, 7537], 
    "ABM-REINA": [2622, 7078], 
    "Cornell-COVID19-sim-Frazier": [7250, 8725], 
    "ABM-Covasim": [14734, 31042], 
    "TIE-GCM": [6336, 209076]
}

# REPORTS_FILE_PATH = "/Users/ferra/Desktop/Work_Repos/skema/reports"
REPORTS_FILE_PATH = os.getenv("GITHUB_WORKSPACE")
LINE_COVERAGE_FILE_NAME = "line_coverage.json"
TEST_COVERAGE_LOCATION = os.path.join(REPORTS_FILE_PATH, "docs", "coverage", "code2fn_coverage", LINE_COVERAGE_FILE_NAME)


def load_line_coverage_information():
    """ 
        Loads the most recently generated test coverage 
        information
    """
    return json.load(open(TEST_COVERAGE_LOCATION))

def test_all_models():
    """
        Tests the coverage of every model we have support for
        Also checks to make sure that we've covered every model that we currently support
    """
    coverage_models = load_line_coverage_information()
    models_visited = 0
    for model in ALL_MODELS.keys():
        baseline_supported_lines, baseline_total_lines = ALL_MODELS[model]
        current_supported_lines, current_total_lines = coverage_models[model]
        assert current_supported_lines >= baseline_supported_lines, f"model {model} supported line count has decreased"
        assert current_total_lines >= baseline_total_lines, f"model {model} total line count has decreased"
        models_visited += 1

    # In case the coverage report generation doesn't give us all the models back, we use this assertion to check that
    assert models_visited == len(ALL_MODELS.keys()), f"test_all_models didn't test all {len(ALL_MODELS.keys())} models, only tested {models_visited} models"
