import torch, math
from collections import Counter


class CreateVocab(object):
    """
    building vocab for the dataset
    stoi: string to index dictionary
    itos: index to string dictionary
    """

    def __init__(self, train, special_tokens=None, min_freq=1):
        self.counter = Counter()
        for line in train["EQUATION"]:
            self.counter.update(line.split())

        self.min_freq = min_freq
        self.tok2ind = dict()
        self.ind2tok = dict()

        # appending special_tokens
        self.s_count = 0
        if special_tokens is not None:
            for i in special_tokens:
                self.tok2ind[i] = self.s_count
                self.ind2tok[self.s_count] = i
                self.s_count += 1

        # appending rest of the vocab
        self.tok_array = [
            tok for (tok, freq) in self.counter.items() if freq >= min_freq
        ]
        self.stoi, self.itos = self.vocab()

    def vocab(self):
        _count = self.s_count
        for t in self.tok_array:
            self.tok2ind[t] = _count
            self.ind2tok[_count] = t
            _count += 1
        return self.tok2ind, self.ind2tok

    def __getitem__(self):
        return self.stoi, self.itos

    def __len__(self):
        return len(self.tok2ind)


# def masking_pad_token(output, mml):
#     """
#     mask will be created using target sequences
#     which then be applied on the model's output seq
#
#     params:
#     output: model's output of shape (seq_len/max_len, B, output_dim)
#     mml: target eqns (seq_len/max_len, B)
#
#     return:
#     output: masked output (len*B, output_dim)
#     mml: masked_mml (len*B)
#     """
#     # masking
#     padding = torch.ones_like(mml) * 0  # 0 is pad_token index
#     mask = (mml != padding)             # [B, l]
#     mml = mml.masked_select(mask)       # [B * (l - len(tok!=<pad>))]
#
#     output_dim = output.shape[-1]
#     output = output.masked_select(mask.unsqueeze(2).expand(-1, -1, output_dim))
#     output = output.contiguous().view(-1, output_dim)
#
#     assert output.shape[0] == mml.shape[0], "output and target shapes are diffrent."
#
#     return output, mml


def garbage2pad(preds, vocab, is_test=False):
    """
    all garbage tokens will be converted to <pad> token
    "garbage" tokens: tokens after <eos> token

    params:
    pred: predicted eqns (B, seq_len/max_len)

    return:
    pred: cleaned pred eqn
    """

    pad_idx = vocab.stoi["<pad>"]
    eos_idx = vocab.stoi["<eos>"]
    for b in range(preds.shape[0]):
        try:
            # cleaning pred
            eos_pos = (preds[b, :] == eos_idx).nonzero(as_tuple=False)[0]
            preds[b, :] = preds[b, : eos_pos + 1]  # pad_idx
        except:
            pass

    return preds


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def calculate_loss(output, mml, vocab):
    """
    calculate Cross Entropy loss
    """
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])
    loss = criterion(output, mml)
    return loss


def calculating_accuracy(pred, mml):
    """
    calculate accuracy

    params:
    pred and mml: (B, l)
    """
    train_acc = torch.sum(pred == mml)
    return train_acc


def beam_search(data, k, alpha, min_length):
    """
    predicting k best possible sequences
    using beam search

    params:
    data: (1, seq_len, output_dim)
    k: beam search parameter
    alpha: degree of regularization in length_normalization
    min_length: param for length_normalization
    """

    # data: (maxlen, output_dim)
    sequences = [[list(), 0.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        for i in range(len(sequences)):
            seq, score = sequences[i]
            log_row = row.log()
            for j in range(len(row)):
                # candidate = [seq + [j], score - math.log(row[j])]
                candidate = [seq + [j], score - log_row[j]]
                all_candidates.append(candidate)

            # order all candiadates by score
            ordered = sorted(all_candidates, key=lambda t: t[1])
            sequences = ordered[:1]
    return sequences


def length_normalization(sequence_length, alpha, min_length):
    ln = (1 + sequence_length) ** alpha / (1 + min_length) ** alpha
    return ln
