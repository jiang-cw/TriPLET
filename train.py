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
train_list = create_list(opt.data_path)
val_list = create_list(opt.val_path)
# SECT_list = create_SECT_list(opt.val_path)

for i in range(opt.increase_factor_data):  # augment the data list for training

    train_list.extend(train_list)
#     val_list.extend(val_list)

print('Number of training patches per epoch:', len(train_list))
print('Number of validation patches per epoch:', len(val_list))
# -------------------------------------




# -----  Transformation and Augmentation process for the data  -----
trainTransforms = [
            # NiftiDataset.Resample(opt.new_resolution, opt.resample),
            # NiftiDataset.Augmentation(),
            # NiftiDataset.Padding((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
            NiftiDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
            ]

train_set = NifitDataSet(train_list, direction=opt.direction, transforms=trainTransforms, train=True)    # define the dataset and loader
train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)  # Here are then fed to the network with a defined batch size
# -------------------------------------





# -----  Creating the Generator and discriminator -----
DenNet = Denoising_Net()
RecNet =  Reconstruction_Net()
AdvNet = PixelDiscriminator(opt)
check_dir(opt.checkpoints_dir)

# -----  Pretrain the Generator and discriminator -----
if opt.resume:
    DenNet.load_state_dict(new_state_dict(opt.DenNetWeights))
    RecNet.load_state_dict(new_state_dict(opt.RecNetWeights))
    AdvNet.load_state_dict(new_state_dict(opt.discriminatorWeights))
    print('Generator and discriminator Weights are loaded')

# -------------------------------------





L_P = nn.MSELoss()  # nn.MSELoss()
L_F = FocalFrequencyLoss()
L_I = GANLoss()
criterionMSE = nn.MSELoss()  # nn.MSELoss()
# -----  Use Single GPU or Multiple GPUs -----
if (opt.gpu_ids != -1) & torch.cuda.is_available():
    use_gpu = True
    DenNet.cuda()
    RecNet.cuda()
    AdvNet.cuda()
    L_P.cuda()
    L_F.cuda()
    L_I.cuda()
    criterionMSE.cuda()

    if num_gpus > 1:
        DenNet = nn.DataParallel(DenNet)
        RecNet = nn.DataParallel(RecNet)
        discriminator = nn.DataParallel(AdvNet)

optim_DenNet = optim.Adam(DenNet.parameters(), betas=(0.5,0.999), lr=opt.DenNetLR)
optim_RecNet = optim.Adam(RecNet.parameters(), betas=(0.5,0.999), lr=opt.RecNetLR)
optim_discriminator = optim.Adam(discriminator.parameters(), betas=(0.5,0.999), lr=opt.discriminatorLR)
net_DenNet_scheduler = get_scheduler(optim_DenNet, opt)
net_RecNet_scheduler = get_scheduler(optim_RecNet, opt)
net_d_scheduler = get_scheduler(optim_discriminator, opt)
# -------------------------------------


# -----  Training Cycle -----
print('Start training :) ')


log_name = opt.task + '_' + opt.netD
print("log_name: ", log_name)
f = open(os.path.join('result/' + opt.task + '/', log_name + ".txt"), "w")

# GradNorm
initial_data, initial_label, _ = next(iter(train_loader))
if use_gpu:
    initial_data = initial_data.cuda()
    initial_label = initial_label.cuda()

initial_fake_b = RecNet(DenNet(initial_data))

initial_inputs = (initial_fake_b, initial_label)
initial_losses = [
    lambda x, y: L_I(AdvNet(torch.cat((initial_data, x), 1)), True),
    lambda x, y: L_F(x, y),
    lambda x, y: L_P(x, y) 
]
initial_grad_norms = calculate_initial_grad_norms(DenNet, initial_losses, initial_inputs)
alpha = [1.0, 1.0, 1.0]  
w = 1.5  

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    mean_generator_total_loss = 0.0
    mean_discriminator_loss = 0.0

    for batch_idx, (data, label, filename) in enumerate(train_loader):
  

        real_a = data
        real_b = label
        real_a = real_a.cuda()



        if use_gpu:                              # forward
            real_b = real_b.cuda()
            # fake_b = DenNet(real_a.cuda())   # generate fake data
            # fake_b = RecNet(fake_b.cuda())

     
            a_np = real_a.squeeze(0).squeeze(0).cpu().numpy() 


            theta = np.linspace(0., 180., max(real_a.shape[2:]), endpoint=False)  
            sinograms = [to_sinogram(a_np[i]) for i in range(a_np.shape[0])]
            sinograms_np = np.stack(sinograms, axis=0)  
            real_a_sinogram = torch.tensor(sinograms_np).unsqueeze(0).unsqueeze(0).float().cuda()  

            b_np = real_b.squeeze(0).squeeze(0).cpu().numpy()  
            sinograms = [to_sinogram(b_np[i]) for i in range(b_np.shape[0])]
            sinograms_np = np.stack(sinograms, axis=0)  
            real_b_sinogram = torch.tensor(sinograms_np).unsqueeze(0).unsqueeze(0).float().cuda()  

   
            fake_b_sinogram = DenNet(real_a_sinogram)


            fake_b_sinogram_np = fake_b_sinogram.detach().squeeze(0).squeeze(0).cpu().numpy()
            reconstructed_slices = [from_sinogram(fake_b_sinogram_np[i]) for i in range(fake_b_sinogram_np.shape[0])]
            reconstructed_np = np.stack(reconstructed_slices, axis=0) 
            fake_b = torch.tensor(reconstructed_np).unsqueeze(0).unsqueeze(0).float().cuda()  

            fake_b = RecNet(fake_b)


            


        

        ######################
        # (1) Update D network
        ######################
        optim_discriminator.zero_grad()

        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = discriminator.forward(fake_ab.detach())
        loss_d_fake = L_I(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = discriminator.forward(real_ab)
        loss_d_real = L_I(pred_real, True)

        # Combined D loss
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        mean_discriminator_loss += discriminator_loss
        discriminator_loss.backward()
        optim_discriminator.step()

        ######################
        # (2) Update G network
        ######################

        optim_DenNet.zero_grad()
        optim_RecNet.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = discriminator.forward(fake_ab)
        loss_g_gan = L_I(pred_fake, True)

        # Second, G(A) = B
        loss_g_l1 = L_F(fake_b_sinogram, real_b_sinogram) 
        loss_g_l2 = L_P(fake_b, real_b) 
       

        task_losses = [loss_g_gan, loss_g_l1, loss_g_l2]
        alpha = grad_norm_update(alpha, task_losses, DenNet, initial_grad_norms, w)

        generator_total_loss = alpha[0] * loss_g_gan + alpha[1] * loss_g_l1 + alpha[2] * loss_g_l2
        # generator_total_loss = criterionMSE(fake_b, real_b)
        mean_generator_total_loss += generator_total_loss

        generator_total_loss.backward()
        optim_DenNet.step()
        optim_RecNet.step()


        ######### Status and display #########
        sys.stdout.write(
            '\r [%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss: %.4f' % (
                epoch, (opt.niter + opt.niter_decay + 1), batch_idx, len(train_loader),
                discriminator_loss, generator_total_loss))
        # print('\r [%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss: %.4f' % (
        #         epoch, (opt.niter + opt.niter_decay + 1), batch_idx, len(train_loader),
        #         discriminator_loss, generator_total_loss), file=f)

    update_learning_rate(net_DenNet_scheduler, optim_DenNet)
    update_learning_rate(optim_RecNet, optim_RecNet)
    update_learning_rate(net_d_scheduler, optim_discriminator)
 


    if epoch % opt.save_fre == 0:
        save_path = os.path.join('result/' + opt.task + '/', 'chk_'+str(epoch))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

     

        ##### Logger ######

        valTransforms = [
            # NiftiDataset.Resample(opt.new_resolution, opt.resample),
            # NiftiDataset.Padding((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
            NiftiDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
        ]

        val_set = NifitDataSet(val_list, direction=opt.direction, transforms=valTransforms, test=True)
        val_loader = DataLoader(val_set, batch_size= 1, shuffle=False, num_workers=opt.workers)


        # test 
        name = list()
        count = 0
 
 
        for batch in val_loader:
            input, target, filename = batch[0].cuda(), batch[1].cuda(), batch[2]
            # name.append([filename[0][0:-7]])

            # prediction = DenNet(input)
            # prediction = RecNet(prediction)

            a_np = input.squeeze(0).squeeze(0).cpu().numpy()  
            sinograms = [to_sinogram(a_np[i]) for i in range(a_np.shape[0])]
            sinograms_np = np.stack(sinograms, axis=0)  
            real_a_sinogram = torch.tensor(sinograms_np).unsqueeze(0).unsqueeze(0).float().cuda()  
            fake_b_sinogram = DenNet(real_a_sinogram)

            fake_b_sinogram_np = fake_b_sinogram.detach().squeeze(0).squeeze(0).cpu().numpy()
            reconstructed_slices = [from_sinogram(fake_b_sinogram_np[i]) for i in range(sinograms_np.shape[0])]
            reconstructed_np = np.stack(reconstructed_slices, axis=0)  
            fake_b = torch.tensor(reconstructed_np).unsqueeze(0).unsqueeze(0).float().cuda()  
            prediction = RecNet(fake_b)
            

            input = input[0,0].cpu().detach().numpy()
            prediction = prediction[0,0].cpu().detach().numpy()
   

            target = target[0,0].cpu().detach().numpy()
            input = (input * 127.5) + 127.5
            prediction = (prediction * 127.5) + 127.5
         
            target = (target * 127.5) + 127.5
            
            if  epoch / opt.save_fre == 1:
                save_result(input, prediction, target, index = filename[0][0:-7], path = save_path)
            else:
                save_result(prediction = prediction, index = filename[0][0:-7], path = save_path)
            count = count + 1       


        torch.save(DenNet.state_dict(), '%s/Den_epoch_{}.pth'.format(epoch) % save_path)
        torch.save(RecNet.state_dict(), '%s/Rec_epoch_{}.pth'.format(epoch) % save_path)
        torch.save(discriminator.state_dict(), '%s/d_epoch_{}.pth'.format(epoch) % save_path)

f.close()




            



       






            



       

