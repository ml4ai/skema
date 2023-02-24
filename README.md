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

- `Dockerfile.skema-py`: Dockerfile for the skema-py service.
- `Dockerfile.skema-rs`: Dockerfile for the skema-rs service.
- `Dockerfile.img2mml`: Dockerfile for the img2mml service.
- `LICENSE.txt`: License for the software components in this repository.
- `README.md`: This README file.
- `data`: Data for testing.
- `docker-compose.skema-py.yml`: Docker Compose file for the skema-py service.
- `docker-compose.skema-rs.yml`: Docker Compose file for the skema-rs service.
- `docker-compose.yml`: Docker Compose file for multiple SKEMA services
  (currently skema-py and skema-rs).
- `docker-compose.memgraph.yml`: Docker Compose file for the memgraph database.
- `docker-compose.img2mml.yml`: Docker Compose file for the img2mml service.
- `docs`: Source code for the project website.
- `notebooks`: Jupyter notebooks for demoing SKEMA functionality.
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

Running the following command in this directory will install the `skema` Python
package into your Python virtual environment (we assume you have one active),
so that it is available for scripts running in that virtual environment.

```
pip install -e .
```

The command above installs the minimum set packages required for the Code2FN
pipeline. There are a couple of extra features that you can install as well
with an alternative invocation. For example, the invocation below installs
packages required for the `moviz` extra.


```
pip install -e .[moviz]
```

For more details on the available extras, see the `pyproject.toml` file.

The `README.md` files in the `skema/skema-rs` and
`skema/text_reading/text_reading` directories provide instructions on how to
run the software components that are written in Rust and Scala respectively.

## Dockerized services

To run the `skema-py` and `skema-rs` services in conjunction, do:

```
docker-compose up --build
```

You can also launch the skema-py and skema-rs services individually by using
their individual `docker-compose.*.yml` files.

To run the img2mml Dockerized service, run

```
docker-compose -f docker-compose.img2mml.yml up --build
```

(make sure the appropriate img2mml model is in the
`skema/img2mml/trained_models` directory - see the `README.md` file in
`skema/img2mml` for details)
