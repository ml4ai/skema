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
        self.xfmer_decoder = decoder
        self.vocab = vocab

    def forward(self, src, device, is_train=False, is_test=False):

        # run the encoder --> get flattened FV of images
        cnn_enc_output = self.cnn_encoder(src)  # (1, L, dec_hid_dim)
        xfmer_enc_output = self.xfmer_encoder(
            cnn_enc_output
        )  # (max_len, 1, dec_hid_dim)

        # inference
        SOS_token = 2
        EOS_token = 3
        max_len = xfmer_enc_output.shape[0]
        trg = torch.tensor([[SOS_token]], dtype=torch.long, device=device)
        for i in range(max_len):
            output = self.xfmer_decoder(trg, xfmer_enc_output, 2, 0)

            top1a = output[i, :, :].argmax(1)

            next_token = torch.tensor([[top1a]], device=device)
            trg = torch.cat((trg, next_token), dim=0)

            # Stop if model predicts end of sentence
            if next_token.view(-1).item() == EOS_token:
                break

        return trg.view(-1).tolist()
