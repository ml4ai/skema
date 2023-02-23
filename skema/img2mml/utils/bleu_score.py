import os
from torchtext.data.metrics import bleu_score

def calculate_bleu_score():
    tt = open("logs/test_targets_100K.txt").readlines()
    tp = open("logs/test_predicted_100K.txt").readlines()
    _tt = open("logs/trimmed_targets.txt", "w")
    _tp = open("logs/trimmed_preds.txt","w")

    for i,j in zip(tt, tp):
        eos_i = i.find("<eos>")
        _tt.write(i[6:eos_i]+"\n")

        eos_j = j.find("<eos>")
        _tp.write(j[6:eos_j]+"\n")

    test = open("logs/trimmed_targets.txt").readlines()
    predicted = open("logs/trimmed_preds.txt").readlines()

    candidate_corpus, references_corpus = [], []

    for t,p in zip(test, predicted):
        candidate_corpus.append(t.split())
        references_corpus.append([p.split()])

    bleu = bleu_score(candidate_corpus, references_corpus)
    return bleu

if __name__ == "__main__":
    print(" calculating Bleu Score...  ")
    tt_bleu = calculate_bleu_score()
    print(' torchtext BLEU score: ', tt_bleu )
