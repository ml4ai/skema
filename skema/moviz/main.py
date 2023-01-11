from flask import Flask, render_template

import os
import shutil
import glob
from pathlib import Path
import argparse

from skema.moviz.utils.create_json import run_cast_to_gromet_pipeline
from skema.moviz.utils.create_viz import draw_graph
from skema.program_analysis.python2cast import python_to_cast


app = Flask(__name__)

@app.route("/")
@app.route("/index")
def execute():

    filepath = Path(__file__).parents[0] / "../../data/gromet/examples/while3/while3.py"
    program_name = filepath.stem

    cast = python_to_cast(str(filepath), cast_obj=True)
    gromet = run_cast_to_gromet_pipeline(cast)

    draw_graph(gromet, program_name)
    full_filename = os.path.join("static", f"{program_name}.png")

    return render_template("index.html", output_image=full_filename)
