import torch
import torch.nn as nn
from elunet_parts import DoubleConv,DownSample,UpSample

class ELUnet(nn.Module):
    def __init__(self,in_channels,out_channels,n:int = 8) -> None:
        """ 
        Construct the Elu-net model.
        Args:
            in_channels: The number of color channels of the input image. 0:for binary 3: for RGB
            out_channels: The number of color channels of the input mask, corresponds to the number
                            of classes.Includes the background
            n: Channels size of the first CNN in the encoder layer. The bigger this value the bigger 
                the number of parameters of the model. Defaults to n = 8, which is recommended by the 
                authors of the paper.
        """
        super().__init__()
        # ------ Input convolution --------------
        self.in_conv = DoubleConv(in_channels,n)
        # -------- Encoder ----------------------
        self.down_1 = DownSample(n,2*n)
        self.down_2 = DownSample(2*n,4*n)
        self.down_3 = DownSample(4*n,8*n)
        self.down_4 = DownSample(8*n,16*n)
        
        # -------- Upsampling ------------------
        self.up_1024_512 = UpSample(16*n,8*n,2)

        self.up_512_64 = UpSample(8*n,n,8)
        self.up_512_128 = UpSample(8*n,2*n,4)
        self.up_512_256 = UpSample(8*n,4*n,2)
        self.up_512_512 = UpSample(8*n,8*n,0)

        self.up_256_64 = UpSample(4*n,n,4)
        self.up_256_128 = UpSample(4*n,2*n,2)
        self.up_256_256 = UpSample(4*n,4*n,0)

        self.up_128_64 = UpSample(2*n,n,2)
        self.up_128_128 = UpSample(2*n,2*n,0)

        self.up_64_64 = UpSample(n,n,0)
     
        # ------ Decoder block ---------------
        self.dec_4 = DoubleConv(2*8*n,8*n)
        self.dec_3 = DoubleConv(3*4*n,4*n)
        self.dec_2 = DoubleConv(4*2*n,2*n)
        self.dec_1 = DoubleConv(5*n,n)
        # ------ Output convolution

        self.out_conv = DoubleConv(n,out_channels)

    def forward(self,x):
        x = self.in_conv(x) # 64
        # ---- Encoder outputs
        x_enc_1 = self.down_1(x) # 128
        x_enc_2 = self.down_2(x_enc_1) # 256
        x_enc_3 = self.down_3(x_enc_2) # 512
        x_enc_4 = self.down_4(x_enc_3) # 1024
    
        # ------ decoder outputs
        x_up_1 = self.up_1024_512(x_enc_4)
        x_dec_4 = self.dec_4(torch.cat([x_up_1,self.up_512_512(x_enc_3)],dim=1))

        x_up_2 = self.up_512_256(x_dec_4)
        x_dec_3 = self.dec_3(torch.cat([x_up_2,
            self.up_512_256(x_enc_3),
            self.up_256_256(x_enc_2)
            ],
        dim=1))

        x_up_3 = self.up_256_128(x_dec_3)
        x_dec_2 = self.dec_2(torch.cat([
            x_up_3,
            self.up_512_128(x_enc_3),
            self.up_256_128(x_enc_2),
            self.up_128_128(x_enc_1)
        ],dim=1))

        x_up_4 = self.up_128_64(x_dec_2)
        x_dec_1 = self.dec_1(torch.cat([
            x_up_4,
            self.up_512_64(x_enc_3),
            self.up_256_64(x_enc_2),
            self.up_128_64(x_enc_1),
            self.up_64_64(x)
        ],dim=1))

        return self.out_conv(x_dec_1)