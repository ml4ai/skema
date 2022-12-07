import torch
import torch.nn as nn
from models.encoding.row_encoding import RowEncoding

class CNN_Encoder(nn.Module):

    def __init__(self, input_channels, dec_hid_dim, dropout, device):
        """
        :param input_channels: input channels of source image
        :param embed_dim: embedding size
        :param hid_dim: size of decoder's RNN
        :param enc_dim: feature size of encoded images
        :param dropout: dropout
        :param device: device to be used
        """
        super(CNN_Encoder, self).__init__()

        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(self.device)
        self.kernel = (3,3)
        self.padding = (1,1)
        self.stride = (1,1)
        self.re = RowEncoding(device, dec_hid_dim, dropout)
        self.linear = nn.Linear(512, dec_hid_dim)

        self.cnn_encoder = nn.Sequential(
                                    # layer 1: [batch, Cin, w, h]
                                    nn.Conv2d(input_channels, 64, kernel_size=self.kernel,
                                                stride=self.stride, padding=self.padding),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),

                                    # layer 2
                                    nn.Conv2d(64, 128, kernel_size=self.kernel,
                                                stride=self.stride, padding=self.padding),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),

                                    # layer 3
                                    nn.Conv2d(128, 256, kernel_size=self.kernel,
                                                stride=self.stride, padding=self.padding),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),

                                    # layer 4: [B, 256, w, h]
                                    nn.Conv2d(256, 256, kernel_size=self.kernel,
                                                stride=self.stride, padding=self.padding),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),

                                    # layer 5
                                    nn.Conv2d(256, 512, kernel_size=self.kernel,
                                                stride=self.stride, padding=self.padding),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),

                                    # layer 6:[B, 512, 10, 33]
                                    nn.Conv2d(512, 512, kernel_size=self.kernel,
                                                stride=self.stride, padding=self.padding),
                                    nn.ReLU()
                                    )
        self.init_weights()

    def init_weights(self):
        '''
        initializing the model wghts with values
        drawn from normal distribution.
        else initialize them with 0.
        '''
        for name, param in self.cnn_encoder.named_parameters():
            if "nn.Conv2d" in name or "nn.Linear" in name:
                if "weight" in name:
                    nn.init.normal_(param.data, mean=0, std=0.1)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0)
            elif "nn.BatchNorm2d" in name:
                if "weight" in name:
                    nn.init.constant_(param.data, 1)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0)


    def forward(self, src, encoding_type=None):
        output = self.cnn_encoder(src)  # (B, 512, W, H)

        if encoding_type == "row_encoding":
            # output: (B, H*W, dec_hid_dim)
            # hidden, cell: [1, B, dec_hid_dim]
            output, hidden, cell = self.re(output)
            return output, hidden, cell

        else:
            output = torch.flatten(output, 2, -1)   # (B, 512, L=H*W)
            output = output.permute(0,2,1)  # (B, L, 512)
            if  encoding_type == "positional_features":
                output += add_positional_features(output) # (B, L, 512)
            return self.linear(output) # (B, L, dec_hid_dim)
