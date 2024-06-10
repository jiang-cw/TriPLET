import sys
import os
import torch
import torch.optim as optim
import torch.nn as nn
from model import *
from discriminators import *
from focal_frequency_loss import *
from utils.NiftiDataset import *
import utils.NiftiDataset as NiftiDataset
from torch.utils.data import DataLoader
from init import Options
from utils.utils import *
import pandas as pd
import numpy as np
from thop import profile
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from skimage.metrics import mean_squared_error, normalized_root_mse, peak_signal_noise_ratio, structural_similarity



# -----  Loading the init options -----
print(torch.cuda.is_available())
print(torch.__version__)  #


opt = Options().parse()
min_pixel = int(opt.min_pixel * ((opt.patch_size[0] * opt.patch_size[1] * opt.patch_size[2]) / 100))

if opt.gpu_ids != '-1':
    num_gpus = len(opt.gpu_ids.split(','))
else:
    num_gpus = 0
print('number of GPU:', num_gpus)
# -------------------------------------



# -----  Loading the list of data -----

test_list = create_list(opt.test_path)


# -----  Creating the Generator and discriminator -----
DenNet = Denoising_Net()
RecNet =  Reconstruction_Net()
AdvNet = PixelDiscriminator(opt)
check_dir(opt.checkpoints_dir)




DenNet.load_state_dict(new_state_dict("result/PET_reconstruction/chk_200/Den_epoch_200.pth"))
RecNet.load_state_dict(new_state_dict("result/PET_reconstruction/chk_200/Rec_epoch_200.pth"))
print('Generator and discriminator Weights are loaded')

if (opt.gpu_ids != -1) & torch.cuda.is_available():
    use_gpu = True
    DenNet.cuda()
    RecNet.cuda()

save_path = os.path.join('result/' + opt.task )
       

testTransforms = [
    # NiftiDataset.Resample(opt.new_resolution, opt.resample),
    # NiftiDataset.Padding((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
    NiftiDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
]

test_set = NifitDataSet(test_list, direction=opt.direction, transforms=testTransforms, test=True)
test_loader = DataLoader(test_set, batch_size= 1, shuffle=False, num_workers=opt.workers)


# test 
name = list()

for batch in test_loader:
    input, target, filename = batch[0].cuda(), batch[1].cuda(), batch[2]

    # prediction = DenNet(input)
    # prediction = RecNet(prediction)

    a_np = input.squeeze(0).squeeze(0).cpu().numpy() 
    theta = np.linspace(0., 180., max(input.shape[2:]), endpoint=False) 
    sinograms = [to_sinogram(a_np[i], theta) for i in range(a_np.shape[0])]
    sinograms_np = np.stack(sinograms, axis=0)  
    real_a_sinogram = torch.tensor(sinograms_np).unsqueeze(0).unsqueeze(0).float().cuda() 


    fake_b_sinogram = DenNet(real_a_sinogram)
 

    fake_b_sinogram_np = fake_b_sinogram.detach().squeeze(0).squeeze(0).cpu().numpy()
    reconstructed_slices = [from_sinogram(fake_b_sinogram_np[i],theta) for i in range(fake_b_sinogram_np.shape[0])]
    reconstructed_np = np.stack(reconstructed_slices, axis=0) 
    fake_b = torch.tensor(reconstructed_np).unsqueeze(0).unsqueeze(0).float().cuda()  
 
    prediction = RecNet(fake_b)
    

    input = input[0,0].cpu().detach().numpy()
    prediction = prediction[0,0].cpu().detach().numpy()
    target = target[0,0].cpu().detach().numpy()

    input = (input * 127.5) + 127.5
    prediction = (prediction * 127.5) + 127.5
    target = (target * 127.5) + 127.5
    
    save_result(input, prediction, target, index = filename[0][0:-7], path = save_path)







    










    



       

