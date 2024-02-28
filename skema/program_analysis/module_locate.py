import argparse
import os
import pkg_resources
import requests
import importlib
import sys
import re
import subprocess
import tempfile
from pathlib import Path

from skema.gromet.fn import TypedValue, ImportSourceType, GrometFNModuleDependencyReference

IMPORT_PATTERN = re.compile(r'^\s*(from\s+[^\s]+\s+import\s+[^\s,]+(?:\s*,\s*[^\s,]+)*|import\s+[^\s,]+(?:\s*,\s*[^\s,]+)*)', re.MULTILINE)

def identify_source_type(source: str):
    if not source:
        return "Unknown"
    if "github" in source:
        return "Repository"
    elif source.startswith("http"):
        return "Url"
    return "Local"
        

def extract_imports(source: str):
    output_references = []

    import_statements = IMPORT_PATTERN.findall(source)
    modules = set(tuple([statement.split()[1] for statement in import_statements]))

    for module in modules:
        source_value = module_locate(module)
        source_type = identify_source_type(source_value)
        output_references.append(
            GrometFNModuleDependencyReference(
                name=module,
                source_reference=TypedValue(
                    type=source_type,
                    value=source_value
        )))

    return output_references

def module_locate(module_name: str) -> str:
    """
    Locates the source of a Python module specified by the import statement.
    If the module is built-in or installed, it returns the file path.
    If the module is on PyPI with a GitHub link, it returns the GitHub URL.
    For PyPI modules, it also attempts to return the tarball URL for the current version.

    :param module_name: The name of the module or submodule as a string.
    :return: The module's file path, GitHub URL, or tarball URL.
    """

    # Attempt to find the module in the local environment
    try:
        module_obj = importlib.import_module(module_name)
        module_file = getattr(module_obj, '__file__', None)
        if module_file:
            module_path = Path(module_file)
            # Check if it's a package
            if module_path.name == "__init__.py":
                return str(module_path.parent)
            return str(module_path)
    except ImportError:
        pass  # If module is not found locally, proceed to check on PyPI

    # Fetch module info from PyPI
    try:
        pypi_url = f"https://pypi.org/pypi/{module_name}/json"
        response = requests.get(pypi_url)
        data = response.json()

        project_urls = data.get('info', {}).get('project_urls', {})
        github_url = project_urls.get('Source', '') or project_urls.get('Homepage', '')
        if 'github.com' in github_url:
            return github_url

        # Get the tarball URL for the current version
        version = data['info']['version']
        releases = data['releases'].get(version, [])
        for release in releases:
            if release['filename'].endswith('.tar.gz'):
                return release['url']
    except Exception as e:
        # Handle errors related to network issues or JSON decoding
        print(f"Error fetching module information from PyPI: {e}")

    return None


"""
# Basic tests 
print(module_locate("import os"))
print(module_locate("import requests"))
print(module_locate("import xml.etree"))
print(module_locate("import minimal"))

# PyDice tests
print(module_locate("import numpy as np"))
print(module_locate("import time"))
print(module_locate("from numba import njit,guvectorize,float64"))
print(module_locate("import scipy.optimize as opt"))
print(module_locate("from matplotlib import pyplot as plt"))
"""




