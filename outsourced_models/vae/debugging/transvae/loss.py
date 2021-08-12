import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

def vae_loss(x, x_out, mu, logvar, true_prop, pred_prop, weights, beta=1):
    "Binary Cross Entropy Loss + Kiebler-Lublach Divergence"
    x = x.long()[:,1:] - 1 #drop the start token
    x = x.contiguous().view(-1) #squeeze into 1 tensor size num_batches*max_seq_len
    x_out = x_out.contiguous().view(-1, x_out.size(2)) # squeeze first and second dims matching above, keeping the 25 class dims.
    BCE = F.cross_entropy(x_out, x, reduction='mean', weight=weights)  #smiles strings have 25 classes or characters (check len(weights))
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if pred_prop is not None:
        MSE = F.mse_loss(pred_prop.squeeze(-1), true_prop)
    else:
        MSE = torch.tensor(0.)
    if torch.isnan(KLD):
        KLD = torch.tensor(0.)
    stopinloss
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

def aae_loss(x, x_out, mu, logvar, true_prop, pred_prop, weights, disc_out, latent_codes, opt, beta=1):
    print("aae loss !")
    
    #formatting x
    x = x.long()[:,1:] - 1 
    x = x.contiguous().view(-1) 
    x_out = x_out.contiguous().view(-1, x_out.size(2))
    
    #generator and autoencoder loss
    opt.g_opt.zero_grad() #zeroing gradients
    
    BCE = F.cross_entropy(x_out, x, reduction='mean', weight=weights) #autoencoder loss
    valid_discriminator_targets = Variable( torch.ones(latent_codes.shape[0], 1), requires_grad=False ) #valid
    disc_out = Variable(disc_out) #using torch autograd Variable
    generator_loss = F.binary_cross_entropy_with_logits(disc_out, valid_discriminator_targets) #discriminator loss vs. valid
    auto_and_gen_loss = BCE + generator_loss
    
    auto_and_gen_loss.backward() #backpropagating
    opt.g_opt.step()
    
    #discriminator loss
    opt.d_opt.zero_grad()#zeroing gradients
    fake_discriminator_targets = Variable( torch.zeros(latent_codes.shape[0], 1), requires_grad=False ) #fake
    disc_generator_loss = F.binary_cross_entropy_with_logits(disc_out, fake_discriminator_targets) #discriminator los vs. fake
 
    discriminator_targets = torch.ones(latent_codes.shape[0], 1)
    discriminator_loss = F.binary_cross_entropy_with_logits(disc_out, discriminator_targets)
    
    disc_loss = 0.5*disc_generator_loss + 0.5*discriminator_loss
    disc_loss.backward() #backpropagating
    opt.d_opt.step() 
    
   
    return auto_and_gen_loss, BCE, torch.tensor(0.), torch.tensor(0.), disc_loss