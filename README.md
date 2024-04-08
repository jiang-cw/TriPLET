# End-to-end Triple-domain PET Enhancement: A Hybrid Denoising-and-reconstruction Framework for Reconstructing Standard-dose PET Images from Low-dose PET Sinograms
This repo contains the supported pytorch code and configuration files of our work


## Overview of our TriPLET
![Overview of our TriPLET) ](assets/fig_framework.pdf)



# System Requirements
This code has been tested on Ubuntu 20.04 and an NVIDIA Tesla A100 GPU. Furthermore it was developed using Python v3.8. 


# Setup
In order to run our model, we suggest you create a virtual environment 
```
conda create -n TriPLET python=3.8
``` 
and activate it with 
```
conda activate TriPLET
```
Subsequently, download and install the required libraries by running 
```
pip install -r requirements.txt
```




# Acknowledgement
This code is heavily build on the following repositories:

(1) https://github.com/lpj-github-io/MWCNNv2

(2) https://github.com/lpj0/MWCNN_PyTorch

(3) https://github.com/microsoft/Swin-Transformer