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
from transvae.opt import NoamOpt, AdamOpt, AAEOpt
from transvae.trans_models import VAEShell, Generator, ConvBottleneck, DeconvBottleneck, PropertyPredictor, Embeddings, LayerNorm

import torch.distributed as dist
import torch.utils.data.distributed

class AAE(VAEShell):
    """
    AAE architecture
    Bypass_bottleneck is set to True and thus the VAE variational or reparaemterization will be avoided
    """
    def __init__(self, params={}, name=None, N=3, d_model=128,
                 d_latent=128, dropout=0.1, tf=True,
                 bypass_bottleneck=True, property_predictor=False,
                 d_pp=256, depth_pp=2, type_pp='deep_net', load_fn=None, discriminator_layers=[640, 256]):
        super().__init__(params, name)

        ### Set learning rate for Adam optimizer
        if 'ADAM_LR' not in self.params.keys():
            self.params['ADAM_LR'] = 3e-4

        ### Store architecture params
        self.model_type = 'aae'
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
        self.params['discriminator_layers'] = discriminator_layers
        self.arch_params = ['N', 'd_model', 'd_latent', 'dropout', 'teacher_force', 'bypass_bottleneck',
                            'property_predictor', 'd_pp', 'depth_pp']

        ### Build model architecture
        if load_fn is None:
            
            if self.params['DDP']:
                ### prepare distributed data parallel (added by Samuel Renaud)
                print("GPUs per node: ",torch.cuda.device_count())
                ngpus_per_node = torch.cuda.device_count()
                
                """ This next line is the key to getting DistributedDataParallel working on SLURM:
                    SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the 
                    current process inside a node and is also 0 or 1 in this example."""
                local_rank = int(os.environ.get("SLURM_LOCALID")) 
                rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank

                """ This next block parses CUDA_VISIBLE_DEVICES to find out which GPUs have been allocated to the job, then sets torch.device to the GPU corresponding       to the local rank (local rank 0 gets the first GPU, local rank 1 gets the second GPU etc) """
                available_gpus = list(os.environ.get('CUDA_VISIBLE_DEVICES').replace(',',""))
                current_device = int(available_gpus[local_rank])
                torch.cuda.set_device(current_device)

                self.build_model()

                """ this block initializes a process group and initiate communications
                        between all processes running on all nodes """
                print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
                #init the process group
                dist.init_process_group(backend=self.params['DIST_BACKEND'], init_method=self.params['INIT_METHOD'],
                                        world_size=self.params['WORLD_SIZE'], rank=rank)
                print("process group ready!")
                print('From Rank: {}, ==> Making model..'.format(rank))

                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[current_device])
            else:
                self.build_model()
        else:
            self.load(load_fn)

    def build_model(self):
        """
        Build model architecture. This function is called during initialization as well as when
        loading a saved model checkpoint
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = RNNEncoder(self.params['d_model'], self.params['d_latent'], self.params['N'],
                             self.params['dropout'], self.params['bypass_bottleneck'], self.device)
        decoder = RNNDecoder(self.params['d_model'], self.params['d_latent'], self.params['N'],
                             self.params['dropout'], 125, self.params['teacher_force'], self.params['bypass_bottleneck'],
                             self.device)
        """ADDING DISCRIMINATOR with proper latent size and number of discriminator layers"""
        discriminator = Discriminator(self.params['d_latent'], self.params['discriminator_layers'])
        
        generator = Generator(self.params['d_model'], self.vocab_size)
        src_embed = Embeddings(self.params['d_model'], self.vocab_size)
        tgt_embed = Embeddings(self.params['d_model'], self.vocab_size)
        if self.params['property_predictor']:
            property_predictor = PropertyPredictor(self.params['d_pp'], self.params['depth_pp'], self.params['d_latent'],
                                                  self.params['type_pp'])
        else:
            property_predictor = None
        self.model = RNNEncoderDecoder(encoder, decoder, discriminator, src_embed, tgt_embed, generator,
                                       property_predictor, self.params)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.model.cuda()
            self.params['CHAR_WEIGHTS'] = self.params['CHAR_WEIGHTS'].cuda()

        ### Initiate optimizers
        #named_parameters returns tuple: (str, params) , store all except discriminator params in 1st opt, store discriminator in 2nd opt
        self.optimizer = AAEOpt(params=[p[1] for p in self.model.named_parameters() 
                                         if (p[1].requires_grad and not "discriminator" in p[0])],
                                disc_params=[p[1] for p in self.model.named_parameters() 
                                         if (p[1].requires_grad and "discriminator" in p[0])],
                                lr=self.params['ADAM_LR'], 
                                generator_optimizer=optim.Adam,
                                discriminator_optimizer=optim.Adam)

        

########## Recurrent Sub-blocks ############

class RNNEncoderDecoder(nn.Module):
    """
    Recurrent Encoder-Decoder Architecture
    """
    def __init__(self, encoder, decoder, discriminator, src_embed, tgt_embed, generator,
                 property_predictor, params):
        super().__init__()
        self.params = params
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.property_predictor = property_predictor

    def forward(self, src, tgt, true_prop, src_mask=None, tgt_mask=None):
        mem, mu, logvar = self.encode(src) # the mem is the latent space from the encoder
        x, h = self.decode(tgt, mem)
        discriminator_outputs = self.discriminator(mem)  #added the discriminator here
        x = self.generator(x)
        if self.property_predictor is not None:
            prop = self.predict_property(mem, true_prop) # the vae bottleneck is bypassed so the "mem" is storing the latent memory
        else:
            prop = None
        return x, mu, logvar, prop, discriminator_outputs, mem

    def encode(self, src):
        return self.encoder(self.src_embed(src))

    def decode(self, tgt, mem):
        return self.decoder(self.src_embed(tgt), mem)

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
        self.z_means = nn.Linear(size, d_latent)
        self.z_var = nn.Linear(size, d_latent)
        self.norm = LayerNorm(size)
        """AAE does not use the std and logvar but will pass through a linear layer that will match the Moses AAE encoder output"""
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

        self.gru = nn.GRU(self.gru_size, self.size, num_layers=N, dropout=dropout)
        self.unbottleneck = nn.Linear(d_latent, size)
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
    
"""
Because of the complexity of the interactions between the outer pipelines and the models I think it would just be easier to implement the AAE here myself by adding to the existing RNN architecture and removing the unnecessary things
"""

class Discriminator(nn.Module):
    def __init__(self, input_size, layers):
        super().__init__()

        in_features = [input_size] + layers
        out_features = layers + [1]

        self.layers_seq = nn.Sequential()
        for k, (i, o) in enumerate(zip(in_features, out_features)):
            self.layers_seq.add_module('linear_{}'.format(k), nn.Linear(i, o))
            if k != len(layers):
                self.layers_seq.add_module('activation_{}'.format(k),
                                           nn.ELU(inplace=True))

    def forward(self, x):
        return self.layers_seq(x)
