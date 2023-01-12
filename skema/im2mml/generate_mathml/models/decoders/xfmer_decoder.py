import torch, math
import torch.nn as nn
from utils.utils import generate_square_subsequent_mask
from models.encoding.positional_encoding_for_xfmer import PositionalEncoding

class Transformer_Decoder(nn.Module):

    def __init__(self, emb_dim, nheads, dec_hid_dim, output_dim, dropout, max_len,
                            n_xfmer_decoder_layers, dim_feedfwd, device):
        super(Transformer_Decoder, self).__init__()
        self.device = device
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.embed = nn.Embedding(output_dim, emb_dim)
        self.pos = PositionalEncoding(emb_dim, dropout, max_len)

        """
        NOTE:
        updated nn.TransformerDecoderLayer doesn't have 'batch_first' argument anymore.
        Therefore, the sequences will be in the shape of (max_len, B)
        """
        xfmer_dec_layer = nn.TransformerDecoderLayer(d_model=dec_hid_dim,
                                                    nhead=nheads,
                                                    dim_feedforward=dim_feedfwd,
                                                    dropout=dropout)

        self.xfmer_decoder = nn.TransformerDecoder(xfmer_dec_layer,
                                            num_layers=n_xfmer_decoder_layers)

        self.modify_dimension = nn.Linear(emb_dim, dec_hid_dim)
        self.final_linear = nn.Linear(dec_hid_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        self.modify_dimension.bias.data.zero_()
        self.modify_dimension.weight.data.uniform_(-0.1, 0.1)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.final_linear.bias.data.zero_()
        self.final_linear.weight.data.uniform_(-0.1, 0.1)

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)

    def forward(self, trg, xfmer_enc_output, sos_idx, pad_idx):
        """
        for inference
        trg: sequnece containing total number of token that has been predicted.
        xfmer_enc_output: input from encoder
        """
        # trg = trg.permute(1,0)  # batch_first --> (len, B)
        sequence_length = trg.shape[0]
        # print("trg:", trg.shape)
        trg_attn_mask = generate_square_subsequent_mask(sequence_length).to(self.device)

        trg = self.embed(trg) * math.sqrt(self.emb_dim)
        pos_trg = self.pos(trg)
        pos_trg = self.modify_dimension(pos_trg)

        # outputs: (max_len-1,B, dec_hid_dim)
        xfmer_dec_outputs = self.xfmer_decoder(tgt=pos_trg,
                                            memory=xfmer_enc_output,
                                            tgt_mask=trg_attn_mask)

        xfmer_dec_outputs = self.final_linear(xfmer_dec_outputs) #(-1,B, output_dim)
        return xfmer_dec_outputs
