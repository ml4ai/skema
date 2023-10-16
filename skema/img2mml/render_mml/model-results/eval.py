import json
import os
import re
import time
from multiprocessing import Pool

import numpy as np
import pylatexenc.latex2text as l2t
import pylatexenc.latexwalker as lw
from sacrebleu import BLEU, corpus_bleu

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
    bleu_scores = np.zeros(len(batch))
    valid_latex = 0
    for i, result in enumerate(batch):
        prediction = result["prediction"]
        ground_truth = result["ground_truth"]
        cur_bleu_score = corpus_bleu(
            hypotheses=[prediction],
            references=[[ground_truth]],
            smooth_method="none",
            tokenize="char",
        ).score
        bleu_scores[i] = cur_bleu_score
        if is_latex_valid(prediction):
            valid_latex += 1
    return bleu_scores, valid_latex


def calculate_test_results(test_results):
    print("Calculating BLEU-4 score...")
    start_time = time.perf_counter()
    num_batches = 4
    batch_size = len(test_results) // num_batches

    batches = [
        list(test_results.values())[i : i + batch_size]
        for i in range(0, len(test_results), batch_size)
    ]

    with Pool(processes=num_batches) as pool:
        score_batches = pool.map(process_batch, batches)

    combined_bleu_scores = np.concatenate([x[0] for x in score_batches])
    combined_valid_latex = sum([x[1] for x in score_batches])
    valid_latex_percentage = combined_valid_latex / len(combined_bleu_scores) * 100

    num_test_results = len(combined_bleu_scores)
    mean_bleu_score = np.mean(combined_bleu_scores)

    finish_time = time.perf_counter()

    print(f"Number of test results: {num_test_results}")
    print(f"BLEU-4 score: {mean_bleu_score:.2f}")
    print(f"Time taken: {finish_time - start_time:.2f}s")
    print(f"Valid LaTeX percentage: {valid_latex_percentage:.2f}%")

    return mean_bleu_score, valid_latex_percentage


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
                load_test_results(os.path.join(results_dir, results_file))
            )
            models[model]["results"][dataset_name] = {
                "bleu_score": results[0],
                "valid_latex_percentage": results[1],
            }

    # write results to result.json
    with open(os.path.join(script_directory, "results.json"), "w") as f:
        json.dump(models, f, indent=4)

    print("--- Results ---")
    for model in models:
        print(f"{model}:")
        print(f"\tRepo: {models[model]['repo_name']}")
        print(f"\tURL: {models[model]['url']}")
        for dataset_name in models[model]["results"]:
            print(
                f"\t{dataset_name} dataset BLEU-4: {models[model]['results'][dataset_name]['bleu_score']:.2f}"
            )
            print(
                f"\t{dataset_name} dataset valid LaTeX percentage: {models[model]['results'][dataset_name]['valid_latex_percentage']:.2f}%"
            )
    print("---------------")
