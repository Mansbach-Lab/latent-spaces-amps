import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

from transvae.tvae_util import *
from transvae.opt import NoamOpt, AdamOpt
from transvae.trans_models import VAEShell, Generator, ConvBottleneck, DeconvBottleneck, PropertyPredictor, Embeddings, LayerNorm

import torch.distributed as dist
import torch.utils.data.distributed
from transvae.DDP import *

import os

# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# Attention architectures inspired by the ^^^ implementation
########## Model Classes ############

class WAE(VAEShell):
    """
    WAE model implementation. (bypasses the VAE reparameterization)
    """
    def __init__(self, params={}, name=None, N=3, d_model=128,
                 d_latent=128, dropout=0.1, tf=True,
                 bypass_bottleneck=True, property_predictor=False,
                 d_pp=256, depth_pp=2, type_pp='deep_net', load_fn=None):
        super().__init__(params, name)
        print("WAE class init called /n")
        ### Set learning rate for Adam optimizer
        if 'ADAM_LR' not in self.params.keys():
            self.params['ADAM_LR'] = 3e-4

        ### Store architecture params
        self.model_type = 'wae'
        self.params['model_type'] = self.model_type
        self.params['N'] = N
        self.params['d_model'] = d_model
        self.params['d_latent'] = d_latent
        self.params['dropout'] = dropout
        self.params['teacher_force'] = tf
        self.params['bypass_bottleneck'] = bypass_bottleneck
        self.params['property_predictor'] = property_predictor
        self.params['type_pp'] = type_pp
        self.params['d_pp'] = d_pp
        self.params['depth_pp'] = depth_pp
        self.arch_params = ['N', 'd_model', 'd_latent', 'dropout', 'teacher_force', 'bypass_bottleneck',
                            'property_predictor', 'd_pp', 'depth_pp']

        ### Build model architecture
        if load_fn is None:
            if self.params['DDP']:
                DDP_init(self)
            else:
                self.build_model()
        else:
            self.load(load_fn)

    def build_model(self):
        print("WAE class build_model called /n")
        """
        Build model architecture. This function is called during initialization as well as when
        loading a saved model checkpoint
        """
        self.device = torch.device("cuda" if 'gpu' in self.params['HARDWARE'] else "cpu")
        encoder = RNNEncoder(self.params['d_model'], self.params['d_latent'], self.params['N'],
                             self.params['dropout'], self.params['bypass_bottleneck'], self.device)
        decoder = RNNDecoder(self.params['d_model'], self.params['d_latent'], self.params['N'],
                             self.params['dropout'], self.tgt_len, self.params['teacher_force'], self.params['bypass_bottleneck'],
                             self.device)
        generator = Generator(self.params['d_model'], self.vocab_size)
        src_embed = Embeddings(self.params['d_model'], self.vocab_size)
        tgt_embed = Embeddings(self.params['d_model'], self.vocab_size)
        if self.params['property_predictor']:
            property_predictor = PropertyPredictor(self.params['d_pp'], self.params['depth_pp'], self.params['d_latent'],
                                                  self.params['type_pp'])
        else:
            property_predictor = None
        self.model = RNNEncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator,
                                       property_predictor, self.params)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if 'gpu' in self.params['HARDWARE']:
            self.model.cuda()
            self.params['CHAR_WEIGHTS'] = self.params['CHAR_WEIGHTS'].cuda()

        ### Initiate optimizer
        self.optimizer = AdamOpt([p for p in self.model.parameters() if p.requires_grad],
                                  self.params['ADAM_LR'], optim.Adam)


########## Recurrent Sub-blocks ############

class RNNEncoderDecoder(nn.Module):
    """
    Recurrent Encoder-Decoder Architecture
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator,
                 property_predictor, params):
        super().__init__()
        self.params = params
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.property_predictor = property_predictor

    def forward(self, src, tgt, true_prop, src_mask=None, tgt_mask=None):
        mem, mu, logvar = self.encode(src)
        x, h = self.decode(tgt, mem)
        x = self.generator(x)
        if self.property_predictor is not None:
            prop = self.predict_property(mem, true_prop) #the vae bottleneck is bypassed so the "mem" is storing the latent memory
        else:
            prop = None
        return x, mu, logvar, prop, mem

    def encode(self, src):
        return self.encoder(self.src_embed(src))

    def decode(self, tgt, mem):
        return self.decoder(self.tgt_embed(tgt), mem)

    def predict_property(self, mem, true_prop):
        return self.property_predictor(mem, true_prop)

class RNNEncoder(nn.Module):
    """
    Simple recurrent encoder architecture
    """
    def __init__(self, size, d_latent, N, dropout, bypass_bottleneck, device):
        super().__init__()
        self.size = size
        self.n_layers = N
        self.bypass_bottleneck = bypass_bottleneck
        self.device = device

        self.gru = nn.GRU(self.size, self.size, num_layers=N, dropout=dropout)
        self.norm = LayerNorm(size)
        """WAE does not use the std and logvar but will pass through a linear layer in the latent space"""
        self.linear_bypass = nn.Linear(size, d_latent)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        h = self.initH(x.shape[0])
        x = x.permute(1, 0, 2)
        x, h = self.gru(x, h)
        mem = self.norm(h[-1,:,:])
        if self.bypass_bottleneck:
            mu, logvar = Variable(torch.tensor([0.0])), Variable(torch.tensor([0.0]))
            mem = self.linear_bypass(mem) #added linear_bypass
        else:
            mu, logvar = self.z_means(mem), self.z_var(mem)
            mem = self.reparameterize(mu, logvar)
        return mem, mu, logvar

    def initH(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.size, device=self.device)

class RNNDecoder(nn.Module):
    """
    Simple recurrent decoder architecture
    """
    def __init__(self, size, d_latent, N, dropout, tgt_length, tf, bypass_bottleneck, device):
        super().__init__()
        self.size = size
        self.n_layers = N
        self.max_length = tgt_length+1
        self.teacher_force = tf
        if self.teacher_force:
            self.gru_size = self.size * 2
        else:
            self.gru_size = self.size
        self.bypass_bottleneck = bypass_bottleneck
        self.device = device
        """WAE does not use the std and logvar but will pass through a linear layer in the latent space"""
        self.linear_bypass = nn.Linear(d_latent, size)
        self.gru = nn.GRU(self.gru_size, self.size, num_layers=N, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(size)

    def forward(self, tgt, mem):
        h = self.initH(mem.shape[0])
        embedded = self.dropout(tgt)
        if not self.bypass_bottleneck:
            mem = F.relu(self.unbottleneck(mem))
            mem = mem.unsqueeze(1).repeat(1, self.max_length, 1)
            mem = self.norm(mem)
        else:
            mem = F.relu(self.linear_bypass(mem)) #added linear_bypass
            mem = mem.unsqueeze(1).repeat(1, self.max_length, 1)
            mem = self.norm(mem)
        if self.teacher_force:
            mem = torch.cat((embedded, mem), dim=2)
        mem = mem.permute(1, 0, 2)
        mem = mem.contiguous()
        x, h = self.gru(mem, h)
        x = x.permute(1, 0, 2)
        x = self.norm(x)
        return x, h

    def initH(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.size, device=self.device)
