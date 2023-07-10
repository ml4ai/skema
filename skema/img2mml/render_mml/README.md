# MathML and LaTeX Comparison Visualization

This folder contains code to visualize the differences between MathML and LaTeX representations of mathematical equations.

## Files Included

- `json_to_js.py`: Converts a JSON file to JS format.
- `latex_data_dev.js`: Contains data for comparing LaTeX and MathML.
- `math_data_dev.js`: Contains data for comparing different methods of generating MathML.
- `images/`: Contains PNG images of LaTeX equations.
- `render_latex_mml.html`: Visualizes the comparison between LaTeX and MathML.
- `render_mathml.html`: Visualizes the comparison between different methods of generating MathML.
- `static/`: Contains files for the webpage's static content.

## Usage

To use this code, simply run `json_to_js.py` on your desired JSON file, and then open either `render_latex_mml.html` or `render_mathml.html` in your browser.

## Text Tweak-er

### Usage

Use this webapp locally to individually view and edit the image annotations returned by Mathpix.

```[bash]
cd text_tweaker
pip install -r requirements.txt # Installs FastAPI
uvicorn main:app
```

The app should now be running. View it on [localhost:8000](http://127.0.0.1:8000/).

### Files in Text Tweaker

- `static/` - Contains the HTML, JS and CSS for th site
- `main.py` - Contains the backend api
- `text_tweaker_annotation_data.json` - The data store that is read and modified by the webapp
