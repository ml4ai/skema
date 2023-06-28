![](http://ci.kraken.sista.arizona.edu/api/badges/ml4ai/skema/status.svg)  
[![Docker lumai/askem-skema-py Image Version (latest by date)](https://img.shields.io/docker/v/lumai/askem-skema-py?sort=date&logo=docker&label=lumai%2Faskem-skema-py)](https://hub.docker.com/r/lumai/askem-skema-py)  
[![Docker lumai/askem-skema-img2mml Image Version (latest by date)](https://img.shields.io/docker/v/lumai/askem-skema-img2mml?sort=date&logo=docker&label=lumai%2Faskem-skema-img2mml)](https://hub.docker.com/r/lumai/askem-skema-img2mml)  
[![Docker lumai/askem-skema-rs Image Version (latest by date)](https://img.shields.io/docker/v/lumai/askem-skema-rs?sort=date&logo=docker&label=lumai%2Faskem-skema-rs)](https://hub.docker.com/r/lumai/askem-skema-rs)  
[![Docker lumai/askem-skema-text-reading Image Version (latest by date)](https://img.shields.io/docker/v/lumai/askem-skema-text-reading?sort=date&logo=docker&label=lumai%2Faskem-skema-text-reading)](https://hub.docker.com/r/lumai/askem-skema-text-reading)

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

- `Dockerfile.skema-py`: Dockerfile for the skema python library (includes program analysis, img2mml, isa, and MOVIZ components).
- `Dockerfile.skema-rs`: Dockerfile for the skema-rs service.
- `LICENSE.txt`: License for the software components in this repository.
- `README.md`: This README file.
- `data`: Data for testing.
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
    - `moviz`: Visualization interface for GroMEt function networks.

Of the Python subpackages, the last three (`skema_py`, `img2mml`, `moviz`) are
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

## Examples

We maintain several containerized examples demonstrating system capabilities at [https://github.com/ml4ai/ASKEM-TA1-DockerVM](https://github.com/ml4ai/ASKEM-TA1-DockerVM).