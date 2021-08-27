import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

class NoamOpt:
    "Optimizer wrapper that implements rate decay (adapted from\
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)"
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size

        self.state_dict = self.optimizer.state_dict()
        self.state_dict['step'] = 0
        self.state_dict['rate'] = 0

    def step(self):
        "Update parameters and rate"
        self.state_dict['step'] += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.state_dict['rate'] = rate
        self.optimizer.step()
        for k, v in self.optimizer.state_dict().items():
            self.state_dict[k] = v

    def rate(self, step=None):
        "Implement 'lrate' above"
        if step is None:
            step = self.state_dict['step']
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def load_state_dict(self, state_dict):
        self.state_dict = state_dict

class AdamOpt:
    "Adam optimizer wrapper"
    def __init__(self, params, lr, optimizer):
        self.optimizer = optimizer(params, lr)
        self.state_dict = self.optimizer.state_dict()

    def step(self):
        self.optimizer.step()
        self.state_dict = self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.state_dict = state_dict

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class AAEOpt:
    "Double Optimizer used for the AAE"
    def __init__(self, params, disc_params, lr, generator_optimizer, discriminator_optimizer):
        self.g_opt = generator_optimizer(params, lr)
        self.d_opt = discriminator_optimizer(disc_params, lr)
        self.state_dict_g = self.g_opt.state_dict()
        self.state_dict_d = self.d_opt.state_dict()
    
    def step_g(self):
        self.g_opt.step()
        self.state_dict_g = self.g_opt.state_dict()
        
    def step_d(self):
        self.d_opt.step()
        self.state_dict_d = self.d_opt.state_dict()
        
    def load_state_dict_g(self, state_dict):
        self.state_dict_g = state_dict
        
    def load_state_dict_d(self, state_dict):
        self.state_dict_d = state_dict
        
    def load_state_dict(self, state_dict_g, state_dict_d):
        self.load_state_dict_g(state_dict_g)
        self.load_state_dict_d(state_dict_d)
        
