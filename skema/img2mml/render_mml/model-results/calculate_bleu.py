import json
import os
import time
from multiprocessing import Pool

import numpy as np
from torchtext.data.metrics import bleu_score
from transformers import PreTrainedTokenizerFast

script_directory = os.path.dirname(os.path.abspath(__file__))
tokenizer_file = os.path.join(script_directory, "lukas_blecher_tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)


def load_test_results(test_results_file):
    with open(test_results_file, "r") as f:
        test_results = json.load(f)
    return test_results


def calculate_bleu_score(candidate_tokens, reference_tokens):
    return bleu_score([candidate_tokens], [[reference_tokens]])


def process_batch(batch):
    bleu_scores = np.zeros(len(batch))
    for i, result in enumerate(batch):
        predicted_tokens = tokenizer.tokenize(result["latex"])
        ground_truth_tokens = tokenizer.tokenize(result["gt_latex"])
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
        "lukas_blecher": {
            "repo_name": "LaTeX-OCR",
            "url": "https://github.com/lukas-blecher/LaTeX-OCR",
        },
        "kingyiusuen": {
            "repo_name": "image-to-latex",
            "url": "https://github.com/kingyiusuen/image-to-latex",
        },
    }
    datasets = ["kingyiusuen", "im2latex100k", "lukas_blecher"]

    for model in models:
        print(f"Calculating {model} results")

        for dataset in datasets:
            print(f"Loading {dataset} test results for {model}...")
            try:
                test_results_path = os.path.join(
                    script_directory, f"{model}/{dataset}_test_results.json"
                )
                models[model][f"{dataset}_set_bleu_score"] = calculate_test_results(
                    load_test_results(test_results_path)
                )
            except FileNotFoundError:
                print(f"Test results for {dataset} not found for {model}")
                print("Skipping...")
                models[model][f"{dataset}_set_bleu_score"] = None

    results = ""
    results += "### Results"
    for model in models:
        results += f"\n\n - {model}"
        results += f"   - **Repo**: {models[model]['repo_name']}"
        results += f"\n   - **URL**: {models[model]['url']}"
        for dataset in datasets:
            if models[model][f"{dataset}_set_bleu_score"] is not None:
                results += f"\n   - **{dataset}** BLEU-4 score: {models[model][f'{dataset}_set_bleu_score']:.2f}"
            else:
                results += f"\n   - **{dataset}** BLEU-4 score: N/A"

    results += "\n\n### Datasets"
    results += "\n\n  - **im2latex100k set**: The processed and cleaned im2latex100k set from https://im2markup.yuntiandeng.com/data/"
    results += f"\n  - **kingyiusuen set**: An extra preprocessed version of the im2latex100k set from the [{models['kingyiusuen']['repo_name']}]({models['kingyiusuen']['url']}) repository"
    results += f"\n  - **im2latex100k set**: The im2latex100k set with custom arxiv and wikipedia additions from the [{models['lukas_blecher']['repo_name']}]({models['lukas_blecher']['url']}) repository"

    with open(os.path.join(script_directory, "results.md"), "w") as f:
        f.write(results)
        print("Results written to results.md")
