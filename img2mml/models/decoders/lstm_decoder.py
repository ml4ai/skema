import torch
import torch.nn as nn
import random
from img2mml.models.attention.bahdanau_attention import BahdanauAttention


class LSTM_Decoder(nn.Module):
    """
    Decoder.
    """

    def __init__(
        self,
        embed_dim,
        encoder_dim,
        dec_hid_dim,
        output_dim,
        n_layers,
        device,
        dropout,
        tfr,
    ):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param hid_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        :param alpha: length_normalization alpha for beam search
        """
        super(LSTM_Decoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.dec_hid_dim = dec_hid_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.device = device
        self.tfr = tfr

        self.attention = BahdanauAttention(
            encoder_dim, dec_hid_dim, n_layers
        )  # attention network

        self.embedding = nn.Embedding(output_dim, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.lstm_input_layer = nn.Linear(embed_dim + dec_hid_dim, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            dec_hid_dim,
            num_layers=n_layers,
            dropout=dropout,
            bias=True,
        )  # decoding LSTMCell
        self.fc = nn.Linear(
            dec_hid_dim, output_dim
        )  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def fine_tune_embeddings(self):
        """
        fine-tuning of embedding layer if not using pretrained embedding layer
        """
        for prameter in self.embedding.parameters():
            parameter.requires_grad = True

    def initialize_hidden(self, batch_size, hid_dim, n_layers):
        """
        initailzing the hidden layer for each and every batch
        as every image is independent to satisfies i.i.d condition
        """
        hidden = torch.zeros(batch_size, hid_dim).unsqueeze(
            0
        )  # [1, batch, hid_dim]
        hidden = hidden.repeat(n_layers, 1, 1)  # (N, batch, dec_hid_dim)
        return hidden

    def forward(
        self,
        enc_output,
        trg,
        is_train,
        is_test,
        encoding_type,
        max_len,
        sos_idx,
    ):
        """
        param enc_output: [B, L, dec_hid_dim]
        param trg: [B, max_len]
        """
        batch_size = trg.shape[0]
        outputs = torch.zeros(max_len, batch_size, self.output_dim).to(
            self.device
        )  # (max_len, B, output_dim)
        hidden = cell = self.initialize_hidden(
            batch_size, self.dec_hid_dim, self.n_layers
        ).to(
            self.device
        )  # (1, B, dec_hid_dim)

        if encoding_type == "row_encoding":
            # enc_output: (B, L, dec_hid_dim)
            # hidden/cell: (1, B, dec_hid_dim)
            enc_output, hidden, cell = enc_output
            if self.n_layers > 1:
                """
                since the number of lstm in row_encoding=1, the hidden shape will always going to be (1,B,dec_hid_dim).
                hence we have to broadcast the hidden and cell layers to match n_layers of lstm in the decoder.
                """
                hidden = hidden.repeat(self.n_layers, 1, 1)
                cell = cell.repeat(self.n_layers, 1, 1)

        token = trg[:, 0]  # (B) , <sos>
        _pred = torch.zeros(trg.shape).to(self.device)
        _pred[:, 0] = torch.full(token.shape, sos_idx)

        for i in range(1, max_len):
            # Embedding
            embeddings = self.embedding(
                token.unsqueeze(0)
            )  # (1, batch_size, embed_dim)
            # Calculate attention

            final_attn_encoding, alpha = self.attention(
                enc_output, hidden
            )  # [ 1, B, dec_hid_dim]

            # lstm input
            lstm_input = torch.cat(
                (embeddings, final_attn_encoding), dim=2
            )  # [1, B, dec_hid_dim+embed]
            lstm_input = self.lstm_input_layer(lstm_input)  # [1, B, embed_dim]
            lstm_output, (hidden, cell) = self.lstm(
                lstm_input, (hidden, cell)
            )  # H: [1(D*N), B, hid]     O: [1, B, Hid*D(=1)]
            predictions = self.fc(lstm_output)  # [1, Batch, output_dim]
            predictions = predictions.squeeze(0)  # (B, output_dim)
            outputs[i] = predictions

            top1 = predictions.argmax(1)  # (B)

            if is_test:
                _pred[:, i] = top1

            if is_train:
                teacher_force = random.random() < self.tfr
                token = trg[:, i] if teacher_force else top1
            else:
                token = top1

        outputs = outputs.permute(1, 0, 2)  # (B, max_len, output_dim)

        return outputs, _pred
