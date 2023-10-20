import json
import os
import re
import time
from multiprocessing import Pool

import numpy as np
import pylatexenc.latex2text as l2t
import pylatexenc.latexwalker as lw
from sacrebleu import BLEU, corpus_bleu
from sacrebleu.tokenizers import tokenizer_13a as t13a

script_directory = os.path.dirname(os.path.abspath(__file__))


def load_test_results(test_results_file):
    with open(test_results_file, "r") as f:
        test_results = json.load(f)
    return test_results


def is_latex_valid(latex_expr: str) -> bool:
    """
    Check if a LaTeX expression is syntactically valid.

    Args:
        latex_expr (str): The LaTeX expression to check.

    Returns:
        bool: True if the expression is valid, False if it contains syntax errors.
    """
    try:
        # Parse the LaTeX expression using LaTeXWalker
        parsed_expr = lw.LatexWalker(latex_expr).get_latex_nodes()

        # Convert the LaTeX expression to text
        text_expr = l2t.latex2text(latex_expr)

        # Check if there are any invalid LaTeX commands or elements
        if parsed_expr:
            return True  # Syntax is valid
        else:
            return False  # Syntax error
    except Exception as e:
        return False  # Syntax error


def process_batch(batch):
    tokenizer = t13a.Tokenizer13a()
    for result in batch:
        prediction = result["prediction"]
        ground_truth = result["ground_truth"]
        cur_bleu_score = corpus_bleu(
            hypotheses=[prediction],
            references=[[ground_truth]],
        ).score
        result["bleu"] = cur_bleu_score
        result["valid_latex"] = 1 if is_latex_valid(prediction) else 0
        result["length"] = len(tokenizer(ground_truth).split(" "))
    return batch


def calculate_test_results(test_results, num_processes=1):
    print("Calculating BLEU-4 score...")
    start_time = time.perf_counter()
    num_batches = num_processes
    batch_size = len(test_results) // num_batches

    batches = [
        list(test_results.values())[i : i + batch_size]
        for i in range(0, len(test_results), batch_size)
    ]

    with Pool(num_batches) as p:
        results = p.map(process_batch, batches)

    final_results = []
    for result in results:
        final_results.extend(result)
    return final_results


if __name__ == "__main__":
    models = {
        "latex-ocr": {
            "repo_name": "LaTeX-OCR",
            "url": "https://github.com/Adi-UA/LaTeX-OCR/tree/main",
            "results": {},
        },
        "image-to-latex": {
            "repo_name": "image-to-latex",
            "url": "https://github.com/Adi-UA/image-to-latex/tree/main",
            "results": {},
        },
        "onmt": {
            "repo_name": "OpenNMT-py",
            "url": "https://github.com/Adi-UA/Open-NMT-1.2.0/tree/main",
            "results": {},
        },
    }

    for model in models:
        print(f"Calculating {model} results")

        results_dir = os.path.join(script_directory, model)
        results_files = os.listdir(results_dir)
        for results_file in results_files:
            if not results_file.endswith(".json"):
                # skip non-json files because they are not test results
                # e.g. .DS_Store
                continue

            dataset_name = results_file.split("_")[0]
            results = calculate_test_results(
                load_test_results(os.path.join(results_dir, results_file)), 4
            )

            result_bins = {
                "0-50": {"bleu": [], "valid_latex": []},
                "50-100": {"bleu": [], "valid_latex": []},
                "100-150": {"bleu": [], "valid_latex": []},
                "150-200": {"bleu": [], "valid_latex": []},
                "200-250": {"bleu": [], "valid_latex": []},
                "250-300": {"bleu": [], "valid_latex": []},
                "300-10000": {"bleu": [], "valid_latex": []},
            }
            # Put results into ins based on length
            for result in results:
                length = result["length"]
                for res_bin in result_bins:
                    l, r = res_bin.split("-")
                    if int(l) <= length < int(r):
                        result_bins[res_bin]["bleu"].append(result["bleu"])
                        result_bins[res_bin]["valid_latex"].append(
                            result["valid_latex"]
                        )
                        break

            # Calculate average BLEU score for each bin
            for res_bin in result_bins:
                mean_bleu_score = (
                    np.round(np.mean(result_bins[res_bin]["bleu"]), 2)
                    if len(result_bins[res_bin]["bleu"]) > 0
                    else 0
                )
                valid_latex_percentage = (
                    (
                        (
                            np.count_nonzero(result_bins[res_bin]["valid_latex"])
                            / len(result_bins[res_bin]["valid_latex"])
                        )
                        * 100
                    )
                    if len(result_bins[res_bin]["valid_latex"]) > 0
                    else 0
                )
                if dataset_name not in models[model]["results"]:
                    models[model]["results"][dataset_name] = {}
                models[model]["results"][dataset_name][res_bin] = {
                    "bleu": mean_bleu_score,
                    "valid_latex": valid_latex_percentage,
                }

    if not os.path.exists(os.path.join(script_directory, "results")):
        os.makedirs(os.path.join(script_directory, "results"))
    with open(os.path.join(script_directory, "results/model-results.json"), "w") as f:
        json.dump(models, f, indent=4)
