# Pytorch implementation of **[ELU-Net: An Efficient and Lightweight U-Net for Medical Image Segmentation](https://ieeexplore.ieee.org/document/9745574)**

## Intro
The authors proposed "an efficient and lightweight
U-Net (ELU-Net) with deep skip connections." with main contributions being:

  * devising a novel ELU-Net to make full use of the full-scale features from the encoder by introducing deep skip connections, which incorporate same and large-scale feature maps of the encoder.
  
## Architecture
![ELU-Net architecture](/img/elunet_arch.png)
![blocks](/img/blocks.png)
![values](/img/ELUnet.drawio.png)

## Usage
``` python
from elunet import ELUnet

# for an RGB input and binary mask output
elunet = ELUnet(3,1,8)

# for an RGB input and 3 channel mask output
elunet = ELUnet(3,3,8)
```
