![](http://ci.kraken.sista.arizona.edu/api/badges/ml4ai/skema/status.svg)

# SKEMA: Scientific Knowledge Extraction and Model Analysis

This is the main code repository for the SKEMA project. It contains the source
code and documentation for the text reading, structural alignment, and model
role analysis components of SKEMA.

## Directory structure

This repository contains code written in Python, Rust, and Scala. The directory
structure has been chosen to make the components written in all these languages
coexist peacefully.

At the top level, we have the following files and directories:

- `Dockerfile.code2fn`: Dockerfile for the Code2FN service.
- `LICENSE.txt`: License for the software components in this repository.
- `README.md`: This README file.
- `data`: Data for testing.
- `docker-compose.code2fn.yml`: Docker Compose file for the Code2FN service.
- `docs`: Source code for the project website.
- `notebooks`: Jupyter notebooks for demoing SKEMA functionality.
- `scripts`: Miscellaneous scripts
- `setup.py`: This `setup.py` file declares and defines the `skema` Python package.
- `skema`

The `skema` directory contains two different types of directories:
- A Rust workspace: `skema-rs`
- A number of Python subpackages:
    - `program_analysis`
    - `gromet`
    - `model_assembly`
    - `text_reading`
    - `code2fn`: Web service for converting code to GroMEt function networks.
    - `im2mml`: Web service for extracting equations from images.
    - `moviz`: Visualization interface for GroMEt function networks.

Of the Python subpackages, the last three (`code2fn`, `im2mml`, `moviz`) are
currently the most 'outward/user-facing' components. The `program_analysis`,
`gromet`, and `model_assembly` directories are comprised primarily of library
code that is used by the `code2fn` service.

The `text_reading` directory contains three subdirectories:
- `mention_linking`: Python subpackage for linking mentions in code and text
- `text_reading`: Scala project for rule-based extraction of mentions of scientific concepts.
- `notebooks`: Jupyter notebooks for demoing text reading/mention linking functionality.

Running the following command in this directory will install the `skema` Python
package into your Python virtual environment (we assume you have one active),
so that it is available for scripts running in that virtual environment.

```
pip install -e .
```

The `README.md` files in the `skema/skema-rs` and
`skema/text_reading/text_reading` directories provide instructions on how to
run the software components that are written in Rust and Scala respectively.

## Dockerized services

To run the Code2FN Dockerized service, run

```
docker-compose -f docker-compose.code2fn.yml up --build
```

To run the Im2MML Dockerized service, run

```
docker-compose -f docker-compose.im2mml.yml up --build
```
