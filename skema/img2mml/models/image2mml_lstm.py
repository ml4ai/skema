import torch
import torch.nn as nn
import random


class Image2MathML_LSTM(nn.Module):
    def __init__(
        self, encoder, decoder, device, encoding_type, max_len, vocab
    ):
        """
        :param encoder: encoder
        :param decoder: decoder
        :param device: device to use for model: cpu or gpu
        """
        super(Image2MathML_LSTM, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.encoding_type = encoding_type
        self.max_len = max_len
        self.vocab = vocab

    def forward(self, src, trg, is_train=False, is_test=False):

        enc_output = self.encoder(
            src, self.encoding_type
        )  # Output: (B, L, dec_hid_dim)

        output, preds = self.decoder(
            enc_output,
            trg,
            is_train,
            is_test,
            self.encoding_type,
            self.max_len,
            self.vocab.stoi["<sos>"],
        )  # (B, max_len, output_dim)

        return output, preds
