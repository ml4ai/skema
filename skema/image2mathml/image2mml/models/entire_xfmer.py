import math
import torch
import torch.nn as nn
from models.encoding.positional_encoding_for_xfmer import PositionalEncoding

class Transformer(nn.Module):

    def __init__(self, output_dim, dec_hid_dim, max_len, nheads, n_encoder_layers,
                    n_decoder_layers, dropout, device):

        super(Transformer, self).__init__()
        self.pos = PositionalEncoding(dec_hid_dim, dropout, max_len)
        self.device=device
        self.output_dim = output_dim
        self.dec_hid_dim = dec_hid_dim
        self.embed = nn.Embedding(output_dim, dec_hid_dim)
        self.xfmer = nn.Transformer(
                                d_model=dec_hid_dim,
                                nhead=nheads,
                                num_encoder_layers=n_encoder_layers,
                                num_decoder_layers=n_decoder_layers,
                                dropout=dropout,
                                )
        self.modify_len = nn.Linear(330, max_len)
        self.out = nn.Linear(dec_hid_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.out.bias.data.zero_()
        self.out.weight.data.uniform_(-0.1, 0.1)


    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, trg, sos_idx, pad_idx, is_test):
        # src : (B, L, dec_hid_dim)
        _preds = torch.zeros(trg.shape).to(self.device)
        _sos_token_shape = trg[:,0].shape

        src = self.modify_len(src.permute(0,2,1)).permute(2,0,1)
        src = src * math.sqrt(self.dec_hid_dim)
        trg_padding_mask = (trg == pad_idx) # (B, max_len)
        trg = trg.permute(1,0)  # (max_len, B)
        trg = self.embed(trg) * math.sqrt(self.dec_hid_dim)
        src = self.pos(src) # (max_len, B, dec_hid_dim)
        trg = self.pos(trg) # (max_len, B, dec_hid_dim)
        src_mask = torch.zeros((src.shape[0], src.shape[0]), device=self.device).type(torch.bool)
        trg_mask = self._generate_square_subsequent_mask(trg.shape[0]).to(self.device)


        xfmer_dec_outputs = self.xfmer(src, trg, src_mask=src_mask, tgt_mask=trg_mask,
                                        tgt_key_padding_mask=trg_padding_mask)
        xfmer_dec_outputs = self.out(xfmer_dec_outputs)

        # preds
        _preds[:,0] = torch.full(_sos_token_shape, sos_idx)
        if is_test:
            for i in range(1,xfmer_dec_outputs.shape[0]):
                top1 = xfmer_dec_outputs[i,:,:].argmax(1)   # (B)
                _preds[:,i] = top1

        return xfmer_dec_outputs.permute(1,0,2), _preds
