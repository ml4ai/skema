import json
import os
import re
import sys

from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
latex_preprocessing_dir = os.path.join(script_dir, "latex_preprocessing")
os.chdir(latex_preprocessing_dir)

input_file_path = sys.argv[1]
input_file_path = os.path.join(script_dir, input_file_path)
temp_file_path = (
    f"{input_file_path}.preprocesed.tmp" if len(sys.argv) < 3 else sys.argv[2]
)
temp_file_path = os.path.join(script_dir, temp_file_path)

print(input_file_path, temp_file_path)


def preprocess_latex():
    print("Starting custom preprocessing...")
    with open(temp_file_path, "r") as f:
        formulas = [line.rstrip() for line in f.readlines()]

    print(f"Loaded {len(formulas)} formulas")
    print("Preprocessing LaTeX formulas...")
    for i, formula in tqdm(enumerate(formulas), total=len(formulas)):
        # Ref: https://github.com/lukas-blecher/LaTeX-OCR/blob/main/pix2tex/dataset/preprocessing/preprocess_formulas.py#L64-L65
        # Replace split, align, etc. with aligned
        formula = re.sub(
            r"\\begin{(split|align|alignedat|alignat|eqnarray)\*?}(.+?)\\end{\1\*?}",
            r"\\begin{aligned}\2\\end{aligned}",
            formula,
            flags=re.S,
        )
        # Replace smallmatrix with matrix
        formula = re.sub(
            r"\\begin{(smallmatrix)\*?}(.+?)\\end{\1\*?}",
            r"\\begin{matrix}\2\\end{matrix}",
            formula,
            flags=re.S,
        )

        # Replace \vspace{...} and \hspace{...} with an empty string
        formula = re.sub(r"\\vspace\s*(\*)?\s*\{[^}]*\}", "", formula)
        formula = re.sub(r"\\hspace\s*(\*)?\s*\{[^}]*\}", "", formula)
        formulas[i] = formula

    with open(temp_file_path, "w") as f:
        f.write("\n".join(formulas))
        f.write("\n")
        print(f"Saved {len(formulas)} formulas")


if __name__ == "__main__":
    output = os.system(
        f"python3 preprocess_formulas.py --mode normalize --input-file {input_file_path} --output-file {temp_file_path}"
    )

    if output != 0:
        print("Error while running latex_preprocessing/preprocess_formulas.py")
        exit(1)
    else:
        preprocess_latex()
        # rename output file to input file if no output file is specified
        if len(sys.argv) < 3:
            os.rename(temp_file_path, input_file_path)
