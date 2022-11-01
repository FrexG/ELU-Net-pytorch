import torch
from elunet_parts import DoubleConv,DownSample,UpSample

class ELUnet(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        # ------ Input convolution --------------
        self.in_conv = DoubleConv(in_channels,64)
        # -------- Encoder ----------------------
        self.down_1 = DownSample(64,128)
        self.down_2 = DownSample(128,256)
        self.down_3 = DownSample(256,512)
        self.down_4 = DownSample(512,1024)
        
        # -------- Upsampling ------------------
        self.up_1024_512 = UpSample(1024,512,2)

        self.up_512_64 = UpSample(512,64,8)
        self.up_512_128 = UpSample(512,128,4)
        self.up_512_256 = UpSample(512,256,2)
        self.up_512_512 = UpSample(512,512,0)

        self.up_256_64 = UpSample(256,64,4)
        self.up_256_128 = UpSample(256,128,2)
        self.up_256_256 = UpSample(256,256,0)

        self.up_128_64 = UpSample(128,64,2)
        self.up_128_128 = UpSample(128,128,0)

        self.up_64_64 = UpSample(64,64,0)
     
        # ------ Decoder block ---------------
        self.dec_4 = DoubleConv(1024,512)
        self.dec_3 = DoubleConv(768,256)
        self.dec_2 = DoubleConv(512,128)
        self.dec_1 = DoubleConv(320,64)
        # ------ Output convolution

        self.out_conv = DoubleConv(64,out_channels)

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