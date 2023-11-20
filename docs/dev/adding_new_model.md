## Add Model to Google Drive
### Original Model Source
The full model clone, including all documentation and non source files should be uploaded to:
```bash
data/models/(climate|ecology|epidemiology|space_weather)/
```
### Zip Archive
A zip archive containing ONLY the source files should be uploaded to:
```bash
data/models/zip-archives/
```
If using a MACOS system, it may be better to automate the generation of this zip archive using a script. See https://github.com/ml4ai/skema/issues/599

## Add model to artifacts.askem.lum.ai bucket
Currently this step is done automatically. Model archives are mirrored once a week to artifcts.askem.lum.ai.  

## Updating models.yaml
Add an entry to ```skema/program_analysis/model_coverage_report/models.yaml``` for the model.
```YAML
Example-Model:
    zip_archive: "https://pathtozip.com"
```