# MOVIZ: Model Visualization

This directory contains the code for the MOVIZ web app for visualizing GroMEt
function networks.

## Installation

Create a [virtual environment](https://docs.python.org/3/library/venv.html) and
install the `skema` package with the `moviz` extra following the instructions
in `../../README.md`.

## Usage

In this directory, run the command `flask --app main run`. The webapp can be
accessed by going to `http://localhost:5000` in your web browser.

**Note:**  The webapp can be started in debug mode with the alternative
invocation below (this is useful for developers).

```
flast --app main --debug run
```
