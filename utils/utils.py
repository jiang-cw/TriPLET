import os
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import morphology
from collections import OrderedDict
from torch.optim import lr_scheduler
import SimpleITK
from skimage.transform import radon, iradon, resize



def save_result(input = None, prediction = None, target = None, index = None, path = None):

    
    # template = image = SimpleITK.ReadImage('./data/2D/DECT/external_test_3/DECT/5219252_1.nii.gz')

    if np.sum(input) != None:
        input = SimpleITK.GetImageFromArray(input)

        # input.SetSpacing(template.GetSpacing())
        # input.SetOrigin(template.GetOrigin())
        # input.SetDirection(template.GetDirection())
        SimpleITK.WriteImage(input, path + '/' + index + '_input.nii.gz')

    if np.sum(prediction) != None:
        prediction = SimpleITK.GetImageFromArray(prediction)

        # prediction.SetSpacing(template.GetSpacing())
        # prediction.SetOrigin(template.GetOrigin())
        # prediction.SetDirection(template.GetDirection())
        SimpleITK.WriteImage(prediction, path + '/' + index + '_prediction.nii.gz')

    if np.sum(target) != None:
        target = SimpleITK.GetImageFromArray(target)

        # target.SetSpacing(template.GetSpacing())
        # target.SetOrigin(template.GetOrigin())
        # target.SetDirection(template.GetDirection())
        SimpleITK.WriteImage(target, path + '/' + index + '_target.nii.gz')


def to_sinogram(slice, theta=None):
    if theta is None:
        theta = np.linspace(0., 180., slice.shape[0], endpoint=False)
    sinogram = radon(slice, theta=theta)
    # sinogram_padded = sinogram[:slice.shape[0], :]
    # sinogram_padded = np.pad(sinogram, ((0, 0), (0, slice.shape[1] - sinogram.shape[1])), mode='constant', constant_values=0)
    return sinogram

def from_sinogram(sinogram, theta=None):
    if theta is None:
        theta = np.linspace(0., 180., sinogram.shape[0], endpoint=False)
    reconstructed_slice = iradon(sinogram, theta=theta, circle=False)
    reconstructed_slice = resize(reconstructed_slice, sinogram.shape, mode='constant')
    return reconstructed_slice

def calculate_initial_grad_norms(model, losses, inputs):
    initial_grad_norms = []
    for loss_fn in losses:
        model.zero_grad()
        loss = loss_fn(*inputs)
        loss.backward(retain_graph=True)
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        initial_grad_norms.append(total_norm)
    return initial_grad_norms


def grad_norm_update(alpha, task_losses, model, initial_grad_norms, w):
    current_grad_norms = []
    for i, loss in enumerate(task_losses):
        model.zero_grad()
        loss.backward(retain_graph=True)
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        current_grad_norms.append(total_norm)
    
    average_norm = sum(current_grad_norms) / len(current_grad_norms)
    target_grad_norms = [(initial_grad_norms[i] / average_norm) ** w for i in range(len(task_losses))]
    
    for i in range(len(alpha)):
        alpha[i] = alpha[i] * target_grad_norms[i]
    
    alpha_sum = sum(alpha)
    for i in range(len(alpha)):
        alpha[i] = alpha[i] / alpha_sum
    
    return alpha

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    # print('learning rate = %.7f' % lr)


def Cor_CoeLoss(y_pred, y_target):
    x = y_pred
    y = y_target
    x_var = x - torch.mean(x)
    y_var = y - torch.mean(y)
    r_num = torch.sum(x_var * y_var)
    r_den = torch.sqrt(torch.sum(x_var ** 2)) * torch.sqrt(torch.sum(y_var ** 2))
    r = r_num / r_den

    # return 1 - r  # best are 0
    return 1 - r**2 # abslute constrain


def dice_coeff(seg, gt, ratio=0.5):
    """
    function to calculate the dice score
    """
    seg = seg.flatten()
    gt = gt.flatten()
    seg[seg > ratio] = np.float32(1)
    seg[seg < ratio] = np.float32(0)
    dice = float(2 * (gt * seg).sum())/float(gt.sum() + seg.sum())
    return dice


def check_dir(path):              # if folder does not exist, create it
    if not os.path.exists(path):
        os.mkdir(path)


def new_state_dict(file_name):
    state_dict = torch.load(file_name)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:6] == 'module':
            name = k[7:]
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    return new_state_dict
