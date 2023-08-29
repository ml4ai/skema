import json
import os
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
    models = {"latex_ocr_model": {}, "singh_model": {}}

    for model in models:
        print(f"Processing {model}...")
        print(f"Loading formulae test results for {model}...")
        models[model]["formulae_set_bleu_score"] = calculate_test_results(
            load_test_results(
                os.path.join(script_directory, f"{model}/formulae_test_results.json")
            )
        )
        print(f"Loading formulae test results for {model}...")
        models[model]["im2latex100k_set_bleu_score"] = calculate_test_results(
            load_test_results(
                os.path.join(
                    script_directory, f"{model}/im2latex100k_test_results.json"
                )
            )
        )

    print("\n---\n")
    for model in models:
        print(f"Model: {model}")
        print(
            f"Formulae set BLEU-4 score: {models[model]['formulae_set_bleu_score']:.2f}"
        )
        print(
            f"Im2Latex100k BLEU-4 score: {models[model]['im2latex100k_set_bleu_score']:.2f}\n"
        )
    print("\n---\n")
    print("Done!")
