
# Configuring your dev environment

We recommend configuring your local development environment using [`conda`](https://docs.conda.io/en/latest/miniconda.html):

```bash
conda create -n skema python=3.8 -c conda-forge rust=1.70.0 openjdk=11 sbt=1.9.0 nodejs=18.15.0
conda activate skema
pip install -e ".[all]"
# fortran grammar for pa
skema-tree-sitter-build-fortran-grammar
# mathjax deps for img2mml
(cd skema/img2mml/data_generation && npm install)
```