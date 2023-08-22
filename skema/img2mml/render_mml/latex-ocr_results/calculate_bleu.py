import json
import os

from torchtext.data.metrics import bleu_score
from transformers import PreTrainedTokenizerFast

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_FILE = os.path.join(SCRIPT_DIRECTORY, "./tokenizer.json")
TEST_RESULTS_FILE = os.path.join(SCRIPT_DIRECTORY, "./img2latex_100k_test_results.json")

tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE)

with open(TEST_RESULTS_FILE, "r") as f:
    test_results = json.load(f)

candidate_corpus = []
references_corpus = []

# Build candidate and references corpus
for result in test_results:
    predicted_latex = result["latex"]
    ground_truth_latex = result["gt_latex"]

    predicted_tokens = tokenizer.tokenize(predicted_latex)
    ground_truth_tokens = tokenizer.tokenize(ground_truth_latex)

    candidate_corpus.append(predicted_tokens)
    references_corpus.append([ground_truth_tokens])

# Calculate BLEU score
bleu_score = bleu_score(candidate_corpus, references_corpus)
print(f"BLEU score: {bleu_score}")
