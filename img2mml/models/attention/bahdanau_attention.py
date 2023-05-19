import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, n_layers):
        """
        :param enc_hid_dim: hidden dim of encoder
        :param dec_hid_dim: hidden dim of decoder
        """
        super(BahdanauAttention, self).__init__()

        self.attn = nn.Linear(dec_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        self.n_layers = n_layers

    def forward(self, encoder_outputs, hidden):
        # hidden = [1, batch size, dec hid dim]
        # encoder_outputs = [batch size, src_len, dec_hid_dim]
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        if self.n_layers == 1:
            hidden = hidden.repeat(src_len, 1, 1).permute(
                1, 0, 2
            )  # (B,L,dec_hid_dim)
        else:
            """
            we need to repeat the hidden to src_len that will result in hid:(n*L, B, dec_hid_dim)
            Hence we need to repeat encoder_outputs too, to match hidden size
            i.e. hid_src_len-src_len
            """
            hidden = hidden.repeat(src_len, 1, 1).permute(
                1, 0, 2
            )  # (B,n*L,dec_hid_dim)
            encoder_outputs = encoder_outputs.repeat(
                1, self.n_layers, 1
            )  # (B, n*L, dec_hid_dim)

        energy = self.attn(torch.cat((hidden, encoder_outputs), dim=2))
        energy = torch.tanh(energy)  # [batch size, src len, dec hid dim]
        attention = self.v(energy).squeeze(2)  # [batch size, src len]
        alpha = F.softmax(attention, dim=1).unsqueeze(
            0
        )  # [1, batch size, src len]
        weighted = torch.bmm(
            alpha.permute(1, 0, 2), encoder_outputs
        )  # [B, 1, dec_hid_dim]

        return weighted.permute(1, 0, 2), alpha
