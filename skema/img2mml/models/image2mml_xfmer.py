import torch
import torch.nn as nn


class Image2MathML_Xfmer(nn.Module):
    def __init__(self, encoder, decoder, vocab, device):
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
        self.device = device
        
        # ===========================
        self.linear = nn.Linear(128, 200)   # added for no xfmer_necoder

    def forward(
        self,
        src,
        trg,
        is_test=False,
        is_inference=False,
        SOS_token=None,
        EOS_token=None,
        PAD_token=None,
    ):
        # run the encoder --> get flattened FV of images
        # for inference Batch(B)=1
        cnn_enc_output = self.cnn_encoder(src)  # (B, L, dec_hid_dim)
        xfmer_enc_output = self.xfmer_encoder(
            cnn_enc_output
        )  # (max_len, B, dec_hid_dim)

        # =========================
        # xfmer_enc_output = self.linear(cnn_enc_output.permute(0,2,1)).permute(2,0,1)        

        if not is_inference:
            # normal training and testing part
            # we will be using torchtext.vocab object
            # while inference, we will provide them
            SOS_token = self.vocab.stoi["<sos>"]
            EOS_token = self.vocab.stoi["<eos>"]
            PAD_token = self.vocab.stoi["<pad>"]

            xfmer_dec_outputs, preds = self.xfmer_decoder(
                trg, xfmer_enc_output, SOS_token, PAD_token, is_test=is_test,
            )

            return xfmer_dec_outputs, preds

        else:
            # inference
            max_len = xfmer_enc_output.shape[0]
            trg = torch.tensor(
                [[SOS_token]], dtype=torch.long, device=self.device
            )
            for i in range(max_len):
                output = self.xfmer_decoder(
                    trg,
                    xfmer_enc_output,
                    SOS_token,
                    PAD_token,
                    is_inference=is_inference,
                )

                top1a = output[i, :, :].argmax(1)

                next_token = torch.tensor([[top1a]], device=self.device)
                trg = torch.cat((trg, next_token), dim=0)

                # Stop if model predicts end of sentence
                if next_token.view(-1).item() == EOS_token:
                    break

            return trg.view(-1).tolist()
