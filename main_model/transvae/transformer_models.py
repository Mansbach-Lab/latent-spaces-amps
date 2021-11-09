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
from transvae.trans_models import *

import torch.distributed as dist
import torch.utils.data.distributed

import os
####### Encoder, Decoder and Generator ############

class TransVAE(VAEShell):
    """
    Transformer-based VAE class. Between the encoder and decoder is a stochastic
    latent space. "Memory value" matrices are convolved to latent bottleneck and
    deconvolved before being sent to source attention in decoder.
    """
    def __init__(self, params={}, name=None, N=3, d_model=128, d_ff=512,
                 d_latent=128, h=4, dropout=0.1, bypass_bottleneck=False,
                 property_predictor=False, d_pp=256, depth_pp=2, load_fn=None):
        super().__init__(params, name)
        """
        Instatiating a TransVAE object builds the model architecture, data structs
        to store the model parameters and training information and initiates model
        weights. Most params have default options but vocabulary must be provided.

        Arguments:
            params (dict, required): Dictionary with model parameters. Keys must match
                                     those written in this module
            name (str): Name of model (all save and log files will be written with
                        this name)
            N (int): Number of repeat encoder and decoder layers
            d_model (int): Dimensionality of model (embeddings and attention)
            d_ff (int): Dimensionality of feed-forward layers
            d_latent (int): Dimensionality of latent space
            h (int): Number of heads per attention layer
            dropout (float): Rate of dropout
            bypass_bottleneck (bool): If false, model functions as standard autoencoder
            property_predictor (bool): If true, model will predict property from latent memory
            d_pp (int): Dimensionality of property predictor layers
            depth_pp (int): Number of property predictor layers
            load_fn (str): Path to checkpoint file
        """

        ### Store architecture params
        self.model_type = 'transformer'
        self.params['model_type'] = self.model_type
        self.params['N'] = N
        self.params['d_model'] = d_model
        self.params['d_ff'] = d_ff
        self.params['d_latent'] = d_latent
        self.params['h'] = h
        self.params['dropout'] = dropout
        self.params['bypass_bottleneck'] = bypass_bottleneck
        self.params['property_predictor'] = property_predictor
        self.params['d_pp'] = d_pp
        self.params['depth_pp'] = depth_pp
        self.arch_params = ['N', 'd_model', 'd_ff', 'd_latent', 'h', 'dropout', 'bypass_bottleneck',
                            'property_predictor', 'd_pp', 'depth_pp']

        ### Build model architecture
        if load_fn is None:
            if self.params['DDP']:
                ### prepare distributed data parallel (added by Samuel Renaud)
                os.system("echo GPUs per node: {}".format(torch.cuda.device_count()))
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
                os.system("echo From Rank: {}, ==> Initializing Process Group...".format(rank))         
                #init the process group
                dist.init_process_group(backend=self.params['DIST_BACKEND'], init_method=self.params['INIT_METHOD'],
                                        world_size=self.params['WORLD_SIZE'], rank=rank)
                os.system("echo process group ready!")
                os.system('echo From Rank: {}, ==> Making model..'.format(rank))
                os.system("echo final check; ngpus_per_node={},local_rank={},rank={},available_gpus={},current_device={}"
                          .format(ngpus_per_node,local_rank,rank,available_gpus,current_device))
                
                #self.model.cuda()

                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[current_device], find_unused_parameters=False)
            else: self.build_model()
        else:
            self.load(load_fn)

    def build_model(self):
        """
        Build model architecture. This function is called during initialization as well as when
        loading a saved model checkpoint
        """
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.params['h'], self.params['d_model'])
        ff = PositionwiseFeedForward(self.params['d_model'], self.params['d_ff'], self.params['dropout'])
        position = PositionalEncoding(self.params['d_model'], self.params['dropout'])
        encoder = VAEEncoder(EncoderLayer(self.params['d_model'], self.src_len, c(attn), c(ff), self.params['dropout']),
                                          self.params['N'], self.params['d_latent'], self.params['bypass_bottleneck'],
                                          self.params['EPS_SCALE'])
        decoder = VAEDecoder(EncoderLayer(self.params['d_model'], self.src_len, c(attn), c(ff), self.params['dropout']),
                             DecoderLayer(self.params['d_model'], self.tgt_len, c(attn), c(attn), c(ff), self.params['dropout']),
                                          self.params['N'], self.params['d_latent'], self.params['bypass_bottleneck'], 
                                          encoder.conv_bottleneck.conv_list)
        src_embed = nn.Sequential(Embeddings(self.params['d_model'], self.vocab_size), c(position))
        tgt_embed = nn.Sequential(Embeddings(self.params['d_model'], self.vocab_size), c(position))
        generator = Generator(self.params['d_model'], self.vocab_size)
        if self.params['property_predictor']:
            property_predictor = PropertyPredictor(self.params['d_pp'], self.params['depth_pp'], self.params['d_latent'])
        else:
            property_predictor = None
        self.model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator, property_predictor)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.model.cuda()
            self.params['CHAR_WEIGHTS'] = self.params['CHAR_WEIGHTS'].cuda()

        ### Initiate optimizer
        self.optimizer = NoamOpt(self.params['d_model'], self.params['LR_SCALE'], self.params['WARMUP_STEPS'],
                                 torch.optim.Adam(self.model.parameters(), lr=0,
                                 betas=(0.9,0.98), eps=1e-9))

class EncoderDecoder(nn.Module):
    """
    Base transformer Encoder-Decoder architecture
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, property_predictor):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.property_predictor = property_predictor

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and tgt sequences (added output shape from conv for decoder ease)"
        mem, mu, logvar, pred_len = self.encode(src, src_mask)
        x = self.decode(mem, src_mask, tgt, tgt_mask)
        x = self.generator(x)
        if self.property_predictor is not None:
            prop = self.predict_property(mu)
        else:
            prop = None
        return x, mu, logvar, pred_len, prop

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, mem, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), mem, src_mask, tgt_mask)

    def predict_property(self, mu):
        return self.property_predictor(mu)



class VAEEncoder(nn.Module):
    "Base transformer encoder architecture"
    def __init__(self, layer, N, d_latent, bypass_bottleneck, eps_scale):
        super().__init__()
        self.layers = clones(layer, N)
        self.conv_bottleneck = ConvBottleneck(layer.size, layer.src_len)
        self.flat_conv_out = self.conv_bottleneck.conv_list[-1] * self.conv_bottleneck.out_channels
        self.z_means, self.z_var = nn.Linear(self.flat_conv_out, d_latent), nn.Linear(self.flat_conv_out, d_latent)
        self.norm = LayerNorm(layer.size)
        self.predict_len1 = nn.Linear(d_latent, d_latent*2)
        self.predict_len2 = nn.Linear(d_latent*2, d_latent)
        self.d_latent = d_latent
        self.bypass_bottleneck = bypass_bottleneck
        self.eps_scale = eps_scale

    def predict_mask_length(self, mem):
        "Predicts mask length from latent memory so mask can be re-created during inference"
        pred_len = self.predict_len1(mem)
        pred_len = self.predict_len2(pred_len)
        pred_len = F.softmax(pred_len, dim=-1)
        pred_len = torch.topk(pred_len, 1)[1]
        return pred_len

    def reparameterize(self, mu, logvar, eps_scale=1):
        "Stochastic reparameterization"
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) * eps_scale
        return mu + eps*std

    def forward(self, x, mask):
        ### Attention and feedforward layers
        for i, attn_layer in enumerate(self.layers):
            x = attn_layer(x, mask)
        ### Batch normalization
        mem = self.norm(x)
        ### Convolutional Bottleneck
        if self.bypass_bottleneck:
            mu, logvar = Variable(torch.tensor([0.0])), Variable(torch.tensor([0.0]))
        else:
            mem = mem.permute(0, 2, 1)
            mem = self.conv_bottleneck(mem)
            mem = mem.contiguous().view(mem.size(0), -1)
            mu, logvar = self.z_means(mem), self.z_var(mem)
            mem = self.reparameterize(mu, logvar, self.eps_scale)
            pred_len = self.predict_len1(mu)
            pred_len = self.predict_len2(pred_len)
        return mem, mu, logvar, pred_len

    def forward_w_attn(self, x, mask):
        "Forward pass that saves attention weights"
        attn_wts = []
        for i, attn_layer in enumerate(self.layers):
            x, wts = attn_layer(x, mask, return_attn=True)
            attn_wts.append(wts.detach().cpu())
        mem = self.norm(x)
        if self.bypass_bottleneck:
            mu, logvar = Variable(torch.tensor([0.0])), Variable(torch.tensor([0.0]))
        else:
            mem = mem.permute(0, 2, 1)
            mem = self.conv_bottleneck(mem)
            mem = mem.contiguous().view(mem.size(0), -1)
            mu, logvar = self.z_means(mem), self.z_var(mem)
            mem = self.reparameterize(mu, logvar, self.eps_scale)
            pred_len = self.predict_len1(mu)
            pred_len = self.predict_len2(pred_len)
        return mem, mu, logvar, pred_len, attn_wts

class EncoderLayer(nn.Module):
    "Self-attention/feedforward implementation"
    def __init__(self, size, src_len, self_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.src_len = src_len
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(self.size, dropout), 2)

    def forward(self, x, mask, return_attn=False):
        if return_attn:
            attn = self.self_attn(x, x, x, mask, return_attn=True)
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            return self.sublayer[1](x, self.feed_forward), attn
        else:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            return self.sublayer[1](x, self.feed_forward)

class VAEDecoder(nn.Module):
    "Base transformer decoder architecture"
    def __init__(self, encoder_layers, decoder_layers, N, d_latent, bypass_bottleneck, conv_list):
        super().__init__()
        self.final_encodes = clones(encoder_layers, 1)
        self.layers = clones(decoder_layers, N)
        self.norm = LayerNorm(decoder_layers.size)
        self.bypass_bottleneck = bypass_bottleneck
        self.size = decoder_layers.size
        self.tgt_len = decoder_layers.tgt_len
        self.conv_out = conv_list[-1] #take the last outputs shape from the convlution
        # Reshaping memory with deconvolution
        self.deconv_bottleneck = DeconvBottleneck(decoder_layers.size, encoder_layers.src_len, conv_list)
        self.linear = nn.Linear(d_latent, 64*self.conv_out)

    def forward(self, x, mem, src_mask, tgt_mask):
        ### Deconvolutional bottleneck (up-sampling)
        if not self.bypass_bottleneck:
            mem = F.relu(self.linear(mem))
            mem = mem.view(-1, 64, self.conv_out)
            mem = self.deconv_bottleneck(mem)
            mem = mem.permute(0, 2, 1)
        ### Final self-attention layer
        for final_encode in self.final_encodes:
            mem = final_encode(mem, src_mask)
        # Batch normalization
        mem = self.norm(mem)
        ### Source-attention layers
        for i, attn_layer in enumerate(self.layers):
            x = attn_layer(x, mem, mem, src_mask, tgt_mask)
        return self.norm(x)

    def forward_w_attn(self, x, mem, src_mask, tgt_mask):
        "Forward pass that saves attention weights"
        if not self.bypass_bottleneck:
            mem = F.relu(self.linear(mem))
            mem = mem.view(-1, 64, self.conv_out)
            mem = self.deconv_bottleneck(mem)
            mem = mem.permute(0, 2, 1)
        for final_encode in self.final_encodes:
            mem, deconv_wts  = final_encode(mem, src_mask, return_attn=True)
        mem = self.norm(mem)
        src_attn_wts = []
        for i, attn_layer in enumerate(self.layers):
            x, wts = attn_layer(x, mem, mem, src_mask, tgt_mask, return_attn=True)
            src_attn_wts.append(wts.detach().cpu())
        return self.norm(x), [deconv_wts.detach().cpu()], src_attn_wts

class DecoderLayer(nn.Module):
    "Self-attention/source-attention/feedforward implementation"
    def __init__(self, size, tgt_len, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.tgt_len = tgt_len
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(self.size, dropout), 3)

    def forward(self, x, memory_key, memory_val, src_mask, tgt_mask, return_attn=False):
        m_key = memory_key
        m_val = memory_val
        if return_attn:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            src_attn = self.src_attn(x, m_key, m_val, src_mask, return_attn=True)
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m_key, m_val, src_mask))
            return self.sublayer[2](x, self.feed_forward), src_attn
        else:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m_key, m_val, src_mask))
            return self.sublayer[2](x, self.feed_forward)

############## Attention and FeedForward ################

class MultiHeadedAttention(nn.Module):
    "Multihead attention implementation (based on Vaswani et al.)"
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads"
        super().__init__()
        assert d_model % h == 0
        #We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, return_attn=False):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                            for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        if return_attn:
            return self.attn
        else:
            return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Feedforward implementation"
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
