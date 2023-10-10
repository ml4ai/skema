# SKEMA: Scientific Knowledge Extraction and Model Analysis

This is the main code repository for the SKEMA project. It contains the source
code and documentation for the text reading, structural alignment, and model
role analysis components of SKEMA.

[For details, see our project documentation](https://ml4ai.github.io/skema/)

## Directory structure

This repository contains code written in Python, Rust, and Scala. The directory
structure has been chosen to make the components written in all these languages
coexist peacefully.

At the top level, we have the following files and directories:

- `Dockerfile.skema-py`: Dockerfile for the skema python library (includes program analysis, img2mml, and isa components).
- `Dockerfile.skema-rs`: Dockerfile for the skema-rs service.
- `LICENSE.txt`: License for the software components in this repository.
- `README.md`: This README file.
- `scripts`: Miscellaneous scripts
- `pyproject.toml`: This file declares and defines the `skema` Python package.
- `skema`

The `skema` directory contains two different types of directories:
- A Rust workspace: `skema-rs`
- A number of Python subpackages:
    - `program_analysis`
    - `gromet`
    - `model_assembly`
    - `text_reading`
    - `skema_py`: Web service for converting code to GroMEt function networks and pyacsets.
    - `img2mml`: Web service for extracting equations from images.

Of the Python subpackages, the last two (`skema_py` and `img2mml`) are
currently the most 'outward/user-facing' components. The `program_analysis`,
`gromet`, and `model_assembly` directories are comprised primarily of library
code that is used by the `skema-py` service.

The `text_reading` directory contains three subdirectories:
- `mention_linking`: Python subpackage for linking mentions in code and text
- `text_reading`: Scala project for rule-based extraction of mentions of scientific concepts.
- `notebooks`: Jupyter notebooks for demoing text reading/mention linking functionality.

## Python
[For instructions on installing our Python library, please see our developer documentation](https://ml4ai.github.io/skema/dev/env/).

## Other
The `README.md` files in the `skema/skema-rs` and
`skema/text_reading/text_reading` directories provide instructions on how to
run the software components that are written in Rust and Scala respectively.

## Docker
[For information on our releases and published docker images, please see this page](https://ml4ai.github.io/skema/changes/)

## Examples

We maintain several containerized examples demonstrating system capabilities at [https://github.com/ml4ai/ASKEM-TA1-DockerVM](https://github.com/ml4ai/ASKEM-TA1-DockerVM).
