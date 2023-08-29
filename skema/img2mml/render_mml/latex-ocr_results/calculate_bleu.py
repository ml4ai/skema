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
test_results = os.path.join(script_directory, "./results/test_results.json")


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


def main(test_results):
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


if __name__ == "__main__":
    with open(test_results, "r") as f:
        test_results = json.load(f)

    main(test_results)
