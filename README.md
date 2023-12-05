# Pytorch implementation of **[ELU-Net: An Efficient and Lightweight U-Net for Medical Image Segmentation](https://ieeexplore.ieee.org/document/9745574)**

## Intro
The authors proposed "an efficient and lightweight
U-Net (ELU-Net) with deep skip connections." with main contributions being:

  * devising a novel ELU-Net to make full use of the full-scale features from the encoder by introducing deep skip connections, which incorporate same and large-scale feature maps of the encoder.
  
## Architecture
![ELU-Net architecture](/img/elunet_arch.png)
![blocks](/img/blocks.png)
![values](/img/ELUnet.drawio.png)

## Requirements
- `python > 3.10`
- `pytorch > 1.7.`

## Usage
``` python
import torch
from elunet import ELUnet

device = "cuda" if torch.cuda.is_available() else "cpu"
# for an RGB input and a single class output + background 
x = torch.randn(1,3,256,256).to(device) # B,C,W,H
elunet = ELUnet(3,1,8).to(device)
out = elunet(x)
logits = torch.sigmoid(out)

# for an RGB input and 2 class output + background
x = torch.randn(1,3,256,256).to(device) # B,C,W,H
elunet = ELUnet(3,3,8).to(device)
out = elunet(x)
logits = torch.softmax(out,dim=1) # C
# to get grayscale mask
mask = torch.argmax(logits,dim=1,keepdims=True)
```
