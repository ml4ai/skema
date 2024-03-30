import pytest
from unittest.mock import patch
from skema.program_analysis.module_locate import identify_source_type, module_locate  

# Testing identify_source_type
@pytest.mark.parametrize("source,expected_type", [
    ("https://github.com/python/cpython", "Compiled"),
    ("https://github.com/other/repository", "Repository"),
    ("http://example.com", "Url"),
    ("local/path/to/module", "Local"),
    ("", "Unknown"),
])
def test_identify_source_type(source, expected_type):
    assert identify_source_type(source) == expected_type

# Mocking requests.get to test module_locate without actual HTTP requests
@pytest.fixture
def mock_requests_get(mocker):
    mock = mocker.patch('skema.program_analysis.module_locate.requests.get')
    return mock

def test_module_locate_builtin_module():
    assert module_locate("sys") == "https://github.com/python/cpython"

def test_module_locate_from_pypi_with_github_source(mock_requests_get):
    mock_requests_get.return_value.json.return_value = {
        'info': {'version': '1.0.0', 'project_urls': {'Source': 'https://github.com/example/project'}},
        'releases': {'1.0.0': [{'filename': 'example-1.0.0.tar.gz', 'url': 'https://example.com/example-1.0.0.tar.gz'}]}
    }
    assert module_locate("example") == "https://github.com/example/project"

def test_module_locate_from_pypi_with_tarball_url(mock_requests_get):
    mock_requests_get.return_value.json.return_value = {
        'info': {'version': '1.2.3'},
        'releases': {'1.2.3': [{'filename': 'package-1.2.3.tar.gz', 'url': 'https://pypi.org/package-1.2.3.tar.gz'}]}
    }
    assert module_locate("package") == "https://pypi.org/package-1.2.3.tar.gz"

def test_module_locate_not_found(mock_requests_get):
    mock_requests_get.side_effect = Exception("Module not found")
    assert module_locate("nonexistent") is None
