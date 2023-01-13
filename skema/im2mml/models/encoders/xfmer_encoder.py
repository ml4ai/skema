import torch, math
import torch.nn as nn
from skema.im2mml.utils import generate_square_subsequent_mask
from skema.im2mml.models.encoding.positional_encoding_for_xfmer import PositionalEncoding


class Transformer_Encoder(nn.Module):
    def __init__(
        self,
        emb_dim,
        dec_hid_dim,
        nheads,
        dropout,
        device,
        max_len,
        n_xfmer_encoder_layers,
        dim_feedfwd,
    ):

        super(Transformer_Encoder, self).__init__()
        self.dec_hid_dim = dec_hid_dim
        self.device = device
        # self.change_length = nn.Linear(330, max_len)
        self.change_length = nn.Linear(1024, max_len)
        self.pos = PositionalEncoding(dec_hid_dim, dropout, max_len)

        """
        NOTE:
        nn.TransformerDecoderLayer doesn't have 'batch_first' argument anymore.
        Therefore, the sequences will be in the shape of (max_len, B)
        """
        xfmer_enc_layer = nn.TransformerEncoderLayer(
            d_model=dec_hid_dim,
            nhead=nheads,
            dim_feedforward=dim_feedfwd,
            dropout=dropout,
        )

        self.xfmer_encoder = nn.TransformerEncoder(
            xfmer_enc_layer, num_layers=n_xfmer_encoder_layers
        )

    def forward(self, src_from_cnn):

        # src_from_cnn: (B, L, dec_hid_dim)
        # change the L=H*W to max_len
        src_from_cnn = src_from_cnn.permute(0, 2, 1)  # (B, dec_hid_dim, L)
        src_from_cnn = self.change_length(
            src_from_cnn
        )  # (B, dec_hid_dim, max_len)
        src_from_cnn = src_from_cnn.permute(
            2, 0, 1
        )  # (max_len, B, dec_hid_dim)

        # embedding + normalization
        """
        no need to embed as src from cnn already has dec_hid_dim as the 3rd dim
        """
        src_from_cnn *= math.sqrt(
            self.dec_hid_dim
        )  # (max_len, B, dec_hid_dim)

        # adding positoinal encoding
        pos_src = self.pos(src_from_cnn)  # (max_len, B, dec_hid_dim)

        # xfmer encoder
        mask = generate_square_subsequent_mask(pos_src.shape[0]).to(
            self.device
        )
        xfmer_enc_output = self.xfmer_encoder(
            src=pos_src, mask=None
        )  # (max_len, B, dec_hid_dim)

        return xfmer_enc_output
