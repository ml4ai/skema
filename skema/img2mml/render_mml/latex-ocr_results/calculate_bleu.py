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


# Reference: https://tex.stackexchange.com/questions/690572/remove-extra-spaces-and-curly-braces-latex-code-in-string-python-normalize-stri
def normalize_latex(latex):
    normalized_latex = re.sub("([^a-zA-Z]) +", r"\1", latex)
    normalized_latex = re.sub("{{(([^{}]|{[^{}]*})*)}}", r"{\1}", normalized_latex)
    while normalized_latex != latex:
        latex = normalized_latex
        normalized_latex = re.sub("{{(([^{}]|{[^{}]*})*)}}", r"{\1}", latex)
    return latex


candidate_corpus = []
normalized_candidate_corpus = []
references_corpus = []

# Build candidate and references corpus
for result in test_results:
    predicted_latex = result["latex"]
    normalized_predicted_latex = normalize_latex(predicted_latex)
    ground_truth_latex = result["gt_latex"]

    predicted_tokens = tokenizer.tokenize(predicted_latex)
    normalized_predicted_tokens = tokenizer.tokenize(normalized_predicted_latex)
    ground_truth_tokens = tokenizer.tokenize(ground_truth_latex)

    candidate_corpus.append(predicted_tokens)
    normalized_candidate_corpus.append(normalized_predicted_tokens)
    references_corpus.append([ground_truth_tokens])

# Calculate BLEU score
standard_bleu_score = bleu_score(candidate_corpus, references_corpus)
norm_bleu_score = bleu_score(normalized_candidate_corpus, references_corpus)

print(f"BLEU score: {standard_bleu_score}")
print(f"Normalized BLEU score: {norm_bleu_score}")
