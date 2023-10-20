import json
import os
import re
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pylatexenc.latex2text as l2t
import pylatexenc.latexwalker as lw
from sacrebleu import BLEU, corpus_bleu
from sacrebleu.tokenizers import tokenizer_13a as t13a

script_directory = os.path.dirname(os.path.abspath(__file__))
tokenizer = t13a.Tokenizer13a()


def load_test_results(test_results_file):
    with open(test_results_file, "r") as f:
        test_results = json.load(f)
    return test_results


def save_test_results(results):
    if not os.path.exists(os.path.join(script_directory, "results")):
        os.makedirs(os.path.join(script_directory, "results"))
    with open(os.path.join(script_directory, "results/model-results.json"), "w") as f:
        json.dump(results, f, indent=4)


def get_bins():
    return [i for i in range(0, 400, 50)]


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
    """This function is called by each process in the pool. It processes a batch of results.

    Args:
        batch (list): A list of results to process. Each result is a dictionary with the keys "prediction" and "ground_truth".

    Returns:
        list: A list of results. Each result is a dictionary with the keys "prediction", "ground_truth", "bleu", "valid_latex", and "length".
    """
    for i, result in enumerate(batch):
        prediction = result["prediction"]
        ground_truth = result["ground_truth"]
        cur_bleu_score = corpus_bleu(
            hypotheses=[prediction],
            references=[[ground_truth]],
        ).score
        batch[i].update(
            {
                "bleu": cur_bleu_score,
                "valid_latex": 1 if is_latex_valid(prediction) else 0,
                "length": len(tokenizer(ground_truth).split(" ")),
            }
        )
    return batch


def calculate_test_results(test_results, num_processes=1):
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

    end_time = time.perf_counter()
    print(f"Processed {len(final_results)} results in {end_time - start_time} seconds")
    return final_results


def bin_results(results):
    result_bins = {
        f"{i}-{i+50 if i < 350 else 'inf'}": {"bleu": [], "valid_latex": []}
        for i in get_bins()
    }

    # Put results into ins based on length
    for result in results:
        length = result["length"]
        for res_bin in result_bins:
            l, r = map(float, res_bin.split("-"))
            if l <= length < r:
                result_bins[res_bin]["bleu"].append(result["bleu"])
                result_bins[res_bin]["valid_latex"].append(result["valid_latex"])
                break
    return result_bins


def calculate_bin_results(models, binned_results):
    for res_bin in binned_results:
        bleu_values = binned_results[res_bin]["bleu"]
        valid_latex_values = binned_results[res_bin]["valid_latex"]

        mean_bleu_score = np.round(np.mean(bleu_values), 2) if bleu_values else 0
        valid_latex_percentage = (
            np.round(
                (np.count_nonzero(valid_latex_values) / len(valid_latex_values) * 100),
                2,
            )
            if valid_latex_values
            else 0
        )

        model_results = (
            models.setdefault(model, {})
            .setdefault("results", {})
            .setdefault(dataset_name, {})
        )
        model_results[res_bin] = {
            "bleu": mean_bleu_score,
            "valid_latex": valid_latex_percentage,
        }

    return models


def vizualize(binned_results):
    datasets = [
        dataset
        for dataset in binned_results[list(binned_results.keys())[0]]["results"].keys()
    ]

    for dataset in datasets:
        width = 0.25  # the width of the bars
        multiplier = 0
        bins = [f"{i}-{i+50 if i < 350 else 'inf'}" for i in get_bins()]
        bin_locations = np.arange(len(bins))

        fig1, ax = plt.subplots(layout="constrained", figsize=(15, 8))
        fig2, ax2 = plt.subplots(layout="constrained", figsize=(15, 8))
        for model in binned_results:
            results = binned_results[model]["results"]

            bleu_scores = [results[dataset][res_bin]["bleu"] for res_bin in bins]
            valid_latex_percentages = [
                results[dataset][res_bin]["valid_latex"] for res_bin in bins
            ]

            offset = width * multiplier
            bleu_rects = ax.bar(bin_locations + offset, bleu_scores, width, label=model)
            valid_latex_rects = ax2.bar(
                bin_locations + offset, valid_latex_percentages, width, label=model
            )

            ax.bar_label(bleu_rects, padding=3)
            ax2.bar_label(valid_latex_rects, padding=3)

            multiplier += 1

        ax.set_ylabel("BLEU Score")
        ax.set_title(f"BLEU Score by Model and Token Length for {dataset}")
        ax.set_xticks(bin_locations + width, bins)
        ax.set_xlabel("Token Length")
        ax.legend(loc="upper left", ncols=len(binned_results.keys()))
        ax.set_ylim(0, 120)

        ax2.set_ylabel("Valid LaTeX %")
        ax2.set_title(f"Valid LaTeX % by Model and Token Length for {dataset}")
        ax2.set_xticks(bin_locations + width, bins)
        ax2.set_xlabel("Token Length")
        ax2.legend(loc="upper left", ncols=len(binned_results.keys()))
        ax2.set_ylim(0, 120)

        # save
        fig1.savefig(
            os.path.join(
                script_directory, f"results/images/{dataset}-bleu-results.png"
            ),
            dpi=144,
        )
        fig2.savefig(
            os.path.join(
                script_directory, f"results/images/{dataset}-valid-latex-results.png"
            ),
            dpi=144,
        )


if __name__ == "__main__":
    models = json.load(open(os.path.join(script_directory, "models.json"), "r"))

    final_results = {}
    for model in models:
        print(f"Calculating {model} results")

        results_dir = os.path.join(script_directory, model)
        results_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]

        for results_file in results_files:
            dataset_name = results_file.split("_")[0]
            results = calculate_test_results(
                load_test_results(os.path.join(results_dir, results_file)), 4
            )

            binned_results = bin_results(results)
            binned_results = calculate_bin_results(models, binned_results)
            final_results.update(binned_results)

    save_test_results(final_results)
    print("Results saved to results directory")

    print("Vizualizing results")
    vizualize(final_results)
