
# Configuring your dev environment

```bash
conda create -n skema python=3.8 -c conda-forge rust=1.70.0 openjdk=11 sbt=1.9.0
conda activate skema
pip install -e ".[all]"
```