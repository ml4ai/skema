import json
import os
import re

from torchtext.data.metrics import bleu_score
from transformers import PreTrainedTokenizerFast

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_FILE = os.path.join(SCRIPT_DIRECTORY, "./tokenizer.json")
TEST_RESULTS_FILE = os.path.join(SCRIPT_DIRECTORY, "./img2latex_100k_test_results.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE)

with open(TEST_RESULTS_FILE, "r") as f:
    test_results = json.load(f)


def normalize_latex(latex):
    return re.sub(
        "{{(([^{}]|{[^{}]*})*)}}", r"{\1}", re.sub("([^a-zA-Z]) +", r"\1", latex)
    )


candidate_corpus, normalized_candidate_corpus, references_corpus = [], [], []

for result in test_results:
    predicted_latex, ground_truth_latex = result["latex"], result["gt_latex"]
    predicted_tokens, normalized_predicted_tokens, ground_truth_tokens = [
        tokenizer.tokenize(latex)
        for latex in [
            predicted_latex,
            normalize_latex(predicted_latex),
            ground_truth_latex,
        ]
    ]

    candidate_corpus.append(predicted_tokens)
    normalized_candidate_corpus.append(normalized_predicted_tokens)
    references_corpus.append([ground_truth_tokens])

standard_bleu_score = bleu_score(candidate_corpus, references_corpus)
norm_bleu_score = bleu_score(normalized_candidate_corpus, references_corpus)

print(f"Corpus BLEU score: {standard_bleu_score}")
print(f"Normalized Corpus BLEU score: {norm_bleu_score}")
