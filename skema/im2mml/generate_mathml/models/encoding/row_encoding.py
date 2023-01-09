import torch


class RowEncoding(torch.nn.Module):
    def __init__(self, device, dec_hid_dim, dropout):
        super(RowEncoding, self).__init__()
        self.device = device
        self.emb = torch.nn.Embedding(256, 512)
        self.lstm = torch.nn.LSTM(
            512,
            dec_hid_dim,
            num_layers=1,
            dropout=dropout,
            bidirectional=False,
            batch_first=False,
        )

    def forward(self, enc_output):
        # enc_output: (B, 512, W, H)
        # Row encoding
        outputs = []
        for wh in range(0, enc_output.shape[2]):
            # row => [batch, 512, W] since for each row,
            # it becomes a 2d matrix of [512, W] for all batches
            row = enc_output[:, :, wh, :]  # [batch, 512, W]
            row = row.permute(2, 0, 1)  # [W, batch, 512(enc_output)]
            position_vector = (
                torch.Tensor(row.shape[1]).long().fill_(wh).to(self.device)
            )  # [batch]
            # self.emb(pos) ==> [batch, 512]
            lstm_input = torch.cat(
                (self.emb(position_vector).unsqueeze(0), row), dim=0
            )  # [W+1, batch, 512]
            lstm_output, (hidden, cell) = self.lstm(lstm_input)
            # output = [W+1, batch, hid_dimx2]
            # hidden/cell = [2x1, batch, hid_dim]
            # we want the fwd and bckwd directional final layer

            outputs.append(lstm_output.unsqueeze(0))

        final_output = torch.cat(
            outputs, dim=0
        )  # [H, W+1, BATCH, dec_hid_dim]
        # modifying it to [H*W+1, batch, dec_hid_dim]
        final_output = final_output.view(
            final_output.shape[0] * final_output.shape[1],
            final_output.shape[2],
            final_output.shape[3],
        )

        # O:[B, L, dec_hid_dim]     H:[1, B, dec_hid_dim]
        return final_output.permute(1, 0, 2), hidden, cell
