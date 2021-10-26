import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

def vae_loss(x, x_out, mu, logvar, true_prop, pred_prop, weights, self, beta=1):
    "Binary Cross Entropy Loss + Kiebler-Lublach Divergence"
    print(beta)
    x = x.long()[:,1:] - 1 #drop the start token
    x = x.contiguous().view(-1) #squeeze into 1 tensor size num_batches*max_seq_len
    x_out = x_out.contiguous().view(-1, x_out.size(2)) # squeeze first and second dims matching above, keeping the 25 class dims.
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    BCE = F.cross_entropy(x_out, x, reduction='mean', weight=weights)  #smiles strings have 25 classes or characters (check len(weights))

    if pred_prop is not None:
        MSE = F.mse_loss(pred_prop.squeeze(-1), true_prop)
    else:
        MSE = torch.tensor(0.)
    if torch.isnan(KLD):
        KLD = torch.tensor(0.)
    return BCE + KLD + MSE, BCE, KLD, MSE

def trans_vae_loss(x, x_out, mu, logvar, true_len, pred_len, true_prop, pred_prop, weights, beta=1):
    "Binary Cross Entropy Loss + Kiebler-Lublach Divergence + Mask Length Prediction"
    x = x.long()[:,1:] - 1
    x = x.contiguous().view(-1)
    x_out = x_out.contiguous().view(-1, x_out.size(2))
    true_len = true_len.contiguous().view(-1)
    BCEmol = F.cross_entropy(x_out, x, reduction='mean', weight=weights)
    BCEmask = F.cross_entropy(pred_len, true_len, reduction='mean')
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if pred_prop is not None:
        MSE = F.mse_loss(pred_prop.squeeze(-1), true_prop)
    else:
        MSE = torch.tensor(0.)
    if torch.isnan(KLD):
        KLD = torch.tensor(0.)
    return BCEmol + BCEmask + KLD + MSE, BCEmol, BCEmask, KLD, MSE

def aae_loss(x, x_out, mu, logvar, true_prop, pred_prop, weights, self, latent_codes, disc_out, opt, beta=1):
   
    #formatting x
    x = x.long()[:,1:] - 1 
    x = x.contiguous().view(-1) 
    x_out = x_out.contiguous().view(-1, x_out.size(2))

    #generator and autoencoder loss
    opt.g_opt.zero_grad() #zeroing gradients
    BCE = F.cross_entropy(x_out, x, reduction='mean', weight=weights) #autoencoder loss

    valid_discriminator_targets =  Variable(torch.ones(latent_codes.shape[0], 1), requires_grad=False) #valid
    generator_loss = F.binary_cross_entropy_with_logits(disc_out, valid_discriminator_targets) #discriminator loss vs. valid
    auto_and_gen_loss = BCE + generator_loss

    auto_and_gen_loss.backward() #backpropagating
    opt.g_opt.step()

    #discriminator loss
    opt.d_opt.zero_grad()#zeroing gradients
    fake_discriminator_targets = Variable( torch.zeros(latent_codes.shape[0], 1), requires_grad=False ) #fake
    disc_generator_loss = F.binary_cross_entropy_with_logits(self.model.discriminator(latent_codes.detach()), fake_discriminator_targets) 

    discriminator_targets = Variable( torch.ones(latent_codes.shape[0], 1), requires_grad=False ) #valid
    discriminator_loss = F.binary_cross_entropy_with_logits(self.model.discriminator(latent_codes.detach()), discriminator_targets)

    disc_loss = 0.5*disc_generator_loss + 0.5*discriminator_loss
    disc_loss.backward() #backpropagating
    opt.d_opt.step() 

    return auto_and_gen_loss, BCE, torch.tensor(0.), torch.tensor(0.), disc_loss

def wae_loss(x, x_out, mu, logvar, true_prop, pred_prop, weights, latent_codes, beta=1):
    "reconstruction and mmd loss"
    #reconstruction loss
    x = x.long()[:,1:] - 1 
    x = x.contiguous().view(-1)
    x_out = x_out.contiguous().view(-1, x_out.size(2)) 
    BCE = F.cross_entropy(x_out, x, reduction='mean', weight=weights)  #smiles strings have 25 classes or characters (check len(weights))
    
   
    z_tilde = latent_codes
    z_var = 2 #variance of gaussian
    sigma = math.sqrt(2)#sigma (Number): scalar variance of isotropic gaussian prior P(Z). set to sqrt(2)
    z = sigma*torch.randn(latent_codes.shape) #sample gaussian
    
    n = z.size(0)
    mmd = im_kernel_sum(z, z, z_var, exclude_diag=True).div(n*(n-1)) + \
          im_kernel_sum(z_tilde, z_tilde, z_var, exclude_diag=True).div(n*(n-1)) + \
          -im_kernel_sum(z, z_tilde, z_var, exclude_diag=False).div(n*n).mul(2)

    return BCE + mmd, BCE, torch.tensor(0.), torch.tensor(0.), mmd

def im_kernel_sum(z1, z2, z_var, exclude_diag=True):
    "adapted from  https://github.com/1Konny/WAE-pytorch/blob/master/ops.py"
    r"""Calculate sum of sample-wise measures of inverse multiquadratics kernel described in the WAE paper.
    Args:
        z1 (Tensor): batch of samples from a multivariate gaussian distribution \
            with scalar variance of z_var.
        z2 (Tensor): batch of samples from another multivariate gaussian distribution \
            with scalar variance of z_var.
        exclude_diag (bool): whether to exclude diagonal kernel measures before sum it all.
    """
    assert z1.size() == z2.size()
    assert z1.ndimension() == 2

    z_dim = z1.size(1)
    C = 2*z_dim*z_var

    z11 = z1.unsqueeze(1).repeat(1, z2.size(0), 1)
    z22 = z2.unsqueeze(0).repeat(z1.size(0), 1, 1)

    kernel_matrix = C/(1e-9+C+(z11-z22).pow(2).sum(2))
    kernel_sum = kernel_matrix.sum()
    # numerically identical to the formulation. but..
    if exclude_diag:
        kernel_sum -= kernel_matrix.diag().sum()

    return kernel_sum