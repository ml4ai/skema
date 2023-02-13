import torch
import torch.nn as nn

class Image2MathML_Xfmer(nn.Module):
    def __init__(self, encoder, decoder, vocab):
        """
        :param encoder: encoders CNN and XFMER
        :param decoder: decoder
        :param device: device to use for model: cpu or gpu
        """
        super(Image2MathML_Xfmer, self).__init__()


        self.cnn_encoder = encoder["CNN"]
        self.xfmer_encoder = encoder["XFMER"]
        # self.cnn_encoder = encoder
        self.xfmer_decoder = decoder
        self.vocab = vocab

    def forward(self, src, trg, is_train=False, is_test=False):

        # trg: (B,max_len)
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_dim = self.xfmer_decoder.output_dim


        # run the encoder --> get flattened FV of images
        cnn_enc_output = self.cnn_encoder(src) # (B, L, dec_hid_dim)
        xfmer_enc_output = self.xfmer_encoder(cnn_enc_output) # (max_len, B, dec_hid_dim)
        # print("xfmer_enc_output: ", xfmer_enc_output.shape)
        xfmer_dec_outputs, preds = self.xfmer_decoder(trg, xfmer_enc_output,
                                                    self.vocab.stoi["<sos>"],
                                                    self.vocab.stoi["<pad>"],
                                                    is_test=is_test)
        """
        entire xfmer
        """
        # xfmer_dec_outputs, preds = self.xfmer_decoder(cnn_enc_output, trg,
        #                                         self.vocab.stoi["<sos>"],
        #                                         self.vocab.stoi["<pad>"],
        #                                         is_test=is_test)

        # xfmer_enc_output: (B, max_len, output_dim)
        # preds: (B, max_len)
        return xfmer_dec_outputs, preds
