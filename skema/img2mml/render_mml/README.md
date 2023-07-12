# MathML and LaTeX Comparison Visualization

This folder contains:

1. **Mathpix Annotator**: A script to post our images to [Mathpix](https://docs.mathpix.com/#introduction) and retrieve Mathpix's OCR annotations for the images.
2. **Text Tweak-er**: A webapp to visualize the Mathpix annotations and make any manual corrections required.

In general, it should be okay to skip to Text Tweak-er's usage section since the Mathpix Annotator doesn't need to be run unless we are regenerating Mathpix's OCR results.

## Mathpix Annotator

A script to post our images to Mathpix and retrieve Mathpix's OCR annotations for the images. This has already been run once and the results from Mathpix have been stored.

**NOTE:** It should NOT be re-run unless we need Mathpix to annotate our images again. The code that queries Mathpix's API has been commented out and should only be uncommented and run if we really need to.

### Usage

1. Create a file called `config.py` in `mathpix_annotation/` and save your `MATHPIX_API_KEY`.
2. Run `main.ipynb`. The code will reuse the results stored from the previous run instead of re-querying Mathpix. To have it regenerate the results from the API uncomment the relevant sections (ONLY IF NEEDED).

### Generated Files

1. `mathpix_full_results.json`: Raw results from Mathpix.
2. `mathpix_errors.json`: Files for which Mathpix returned errors.
3. `text_tweaker_annotation_data.json`: Trimmed Mathpix results sorted in increasing order of confidence to be used in Text Tweak-er.

### Other Files

1. `main.ipynb`: Driver code for the Mathpix Annotator script.
2. `batchRequestHandler`: A helper class for making Mathpix batch requests.
3. `image_ids.json`: A JSON with the IDs for the images being posted.

## Text Tweak-er

A webapp to visualize the Mathpix annotations and make any manual corrections required.

### Usage

Use this webapp locally to individually view and edit the image annotations returned by Mathpix.

```[bash]
cd text_tweaker
pip install -r requirements.txt # Installs FastAPI
uvicorn main:app # Runs the webapp
```

The app should now be running. View it on [localhost:8000](http://127.0.0.1:8000/).

### Files

- `static/` - Contains the HTML, JS and CSS for the site.
- `main.py` - Contains the backend api.
- `text_tweaker_annotation_data.json` - The data store that is read and modified by the webapp.
