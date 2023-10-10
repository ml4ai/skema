import json
import os
import re
import time
from multiprocessing import Pool

import numpy as np
from torchtext.data.metrics import bleu_score
from transformers import PreTrainedTokenizerFast

script_directory = os.path.dirname(os.path.abspath(__file__))
tokenizer_file = os.path.join(script_directory, "tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)


def load_test_results(test_results_file):
    with open(test_results_file, "r") as f:
        test_results = json.load(f)
    return test_results


def calculate_bleu_score(candidate_tokens, reference_tokens):
    return bleu_score([candidate_tokens], [[reference_tokens]])


def post_process(s: str):
    """Remove unnecessary whitespace from LaTeX code.

    Args:
        s (str): Input string

    Returns:
        str: Processed image
    """
    text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
    letter = "[a-zA-Z]"
    noletter = "[\W_^\d]"
    names = [x[0].replace(" ", "") for x in re.findall(text_reg, s)]
    s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
    news = s
    while True:
        s = news
        news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", s)
        news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
        news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
        if news == s:
            break
    return s


def process_batch(batch):
    bleu_scores = np.zeros(len(batch))
    for i, result in enumerate(batch):
        predicted_tokens = tokenizer.tokenize(post_process(result["prediction"]))
        ground_truth_tokens = tokenizer.tokenize(post_process(result["ground_truth"]))
        cur_bleu_score = calculate_bleu_score(predicted_tokens, ground_truth_tokens)
        bleu_scores[i] = cur_bleu_score
    return bleu_scores


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
        bleu_scores_batches = pool.map(process_batch, batches)

    combined_bleu_scores = np.concatenate(bleu_scores_batches)

    num_test_results = len(combined_bleu_scores)
    mean_bleu_score = np.mean(combined_bleu_scores)

    finish_time = time.perf_counter()

    print(f"Number of test results: {num_test_results}")
    print(f"BLEU-4 score: {mean_bleu_score:.2f}")
    print(f"Time taken: {finish_time - start_time:.2f}s")

    return mean_bleu_score


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
    }

    for model in models:
        print(f"Calculating {model} results")

        results_dir = os.path.join(script_directory, model)
        results_files = os.listdir(results_dir)
        for results_file in results_files:
            dataset_name = results_file.split("_")[0]
            models[model]["results"][dataset_name] = calculate_test_results(
                load_test_results(os.path.join(results_dir, results_file))
            )

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
                f"\t{dataset_name} dataset BLEU-4: {models[model]['results'][dataset_name]:.2f}"
            )
