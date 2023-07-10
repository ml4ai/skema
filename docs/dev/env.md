
# Configuring your dev environment

We recommend configuring your local development environment using [`conda`](https://docs.conda.io/en/latest/miniconda.html):

```bash
conda create -n skema python=3.8 -c conda-forge rust=1.70.0 openjdk=11 sbt=1.9.0 nodejs=18.15.0
conda activate skema
# fortran grammar for pa
python skema/program_analysis/TS2CAST/build_tree_sitter_fortran.py
# download the checkpoint for the img2mml service
curl -L https://artifacts.askem.lum.ai/skema/img2mml/models/cnn_xfmer_arxiv_im2mml_with_fonts_boldface_best.pt > skema/img2mml/trained_models/cnn_xfmer_arxiv_im2mml_with_fonts_boldface_best.pt
# mathjax deps for img2mml
(cd skema/img2mml/data_generation && npm install)
```

## Installing the Python library in development mode

```bash
pip install -e ".[core]"
```

The command above installs the minimum set packages required for the Code2FN pipeline. 

To additionally install dev dependencies:

```bash
pip install -e ".[core,dev]"
```

To install **all** components (including dev dependencies for documentation generation):
```bash
pip install ".[all]"