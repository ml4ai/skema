import json
import os
import re

import numpy as np
from torchtext.data.metrics import bleu_score
from transformers import PreTrainedTokenizerFast

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_FILE = os.path.join(SCRIPT_DIRECTORY, "./tokenizer.json")
TEST_RESULTS_FILE = os.path.join(SCRIPT_DIRECTORY, "./img2latex_100k_test_results.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE)


with open(TEST_RESULTS_FILE, "r") as f:
    test_results = json.load(f)

bleu_scores = np.zeros(len(test_results))
for i, result_id in enumerate(test_results):
    predicted_latex = test_results[result_id]["latex"]
    ground_truth_latex = test_results[result_id]["gt_latex"]

    predicted_tokens = tokenizer.tokenize(predicted_latex)
    ground_truth_tokens = tokenizer.tokenize(ground_truth_latex)

    canidate_corpus = [predicted_tokens]
    references_corpus = [[ground_truth_tokens]]

    cur_bleu_score = bleu_score(canidate_corpus, references_corpus)
    bleu_scores[i] = cur_bleu_score


print(f"Number of test results: {len(bleu_scores)}")
print(f"BLEU-4 score: {np.mean(bleu_scores)}")
