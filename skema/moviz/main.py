from flask import Flask, render_template

import os
from pathlib import Path

from skema.moviz.utils.create_json import run_cast_to_gromet_pipeline
from skema.moviz.utils.create_viz import draw_graph
from skema.program_analysis.python2cast import python_to_cast
import base64


app = Flask(__name__)

def visualize_single_file(filepath) -> str:
    """Returns base64-encoded string representing the Graphviz layout"""
    program_name = filepath.stem
    cast = python_to_cast(str(filepath), cast_obj=True)
    gromet = run_cast_to_gromet_pipeline(cast)
    graph = draw_graph(gromet, program_name)
    print(graph)
    # Get the raw bytes, encode them via base64, then decode them via utf-8.
    # We embed the image directly in the HTML template.
    output = str(base64.b64encode(graph.pipe()), encoding="utf-8")
    return output

@app.route("/")
@app.route("/index")
def execute():

    cwd = Path(__file__).parents[0]
    filepath = cwd / "../../data/gromet/examples/exp2/exp2.py"
    output = visualize_single_file(filepath)
    return render_template("index.html", output_image=output)

@app.route("/visualize/<filename>/")
def visualize_single_python_file(filename):
    """Visualize a single Python file GroMEt. The file must be placed in the
    `inputs` directory."""

    cwd = Path(__file__).parents[0]
    filepath = cwd / "inputs" / filename
    output = visualize_single_file(filepath)
    return render_template("index.html", output_image=output)
