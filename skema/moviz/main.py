from flask import Flask, render_template
import os

from create_json import run_pipeline_export_gromet
from create_viz import draw_graph

app = Flask(__name__)

# OUTPUT_FOLDER = os.path.join('outputs', 'images')
# app.config['UPLOAD_FOLDER'] = 'outputs'

@app.route("/")
@app.route("/index")
def execute():

    PYTHON_SOURCE_FILE = "exp1.py"
    PROGRAM_NAME = PYTHON_SOURCE_FILE.rsplit(".")[0].rsplit("/")[-1]

    run_pipeline_export_gromet(PYTHON_SOURCE_FILE, PROGRAM_NAME)

    draw_graph(PROGRAM_NAME)
    full_filename = os.path.join('static', f"{PROGRAM_NAME}.png")

    return render_template("index.html", output_image = full_filename)