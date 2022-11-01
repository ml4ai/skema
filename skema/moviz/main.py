from flask import Flask, render_template

import os
import shutil
import glob

from utils.create_json import run_pipeline_export_gromet
from utils.create_viz import draw_graph


app = Flask(__name__)


@app.route("/")
@app.route("/index")
def execute():

    PYTHON_SOURCE_FILE = "inputs/while3.py"
    PROGRAM_NAME = PYTHON_SOURCE_FILE.rsplit(".")[0].rsplit("/")[-1]

    run_pipeline_export_gromet(PYTHON_SOURCE_FILE, PROGRAM_NAME)

    src = f"{PROGRAM_NAME}--Gromet-FN-auto.json"
    dest = "data"

    if os.path.exists(
        os.path.join(dest, f"{PROGRAM_NAME}--Gromet-FN-auto.json")
    ):
        os.remove(os.path.join(dest, f"{PROGRAM_NAME}--Gromet-FN-auto.json"))
        shutil.move(src, dest)
    else:
        shutil.move(src, dest)

    draw_graph(PROGRAM_NAME)
    full_filename = os.path.join("static", f"{PROGRAM_NAME}.png")

    # print(full_filename)
    # if os.path.exists(os.path.join(dest, f"{PROGRAM_NAME}--Gromet-FN-auto.json")):
    #     os.remove(os.path.join(dest, f"{PROGRAM_NAME}--Gromet-FN-auto.json"))

    return render_template("index.html", output_image=full_filename)
