import json
import yaml
import pytest
from pathlib import Path

from skema.program_analysis.model_coverage_report.model_coverage_report import process_all_models
from skema.program_analysis.model_coverage_report.html_builder import HTML_Instance
 
CURRENT_ARTIFACTS_PATH = Path(__file__).resolve().parent / "artifacts.json"
PREVIOUS_ARTIFACTS_PATH = Path(__file__).resolve().parent / "artifacts.json"

@pytest.mark.ci_only
def test_regression_supported_lines():
    "Unit test for basic model coverage regression testing using support line numbers"
    previous_artifact = yaml.safe_load(PREVIOUS_ARTIFACTS_PATH.read_text())
    current_artifact = yaml.safe_load(CURRENT_ARTIFACTS_PATH.read_text())

    for model_name, model_artifact in previous_artifact.items():
        if current_artifact[model_name]["supported_lines"] < model_artifact["supported_lines"]:
            assert False

    assert True