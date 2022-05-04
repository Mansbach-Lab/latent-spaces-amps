import os
import pickle
import pkg_resources

import numpy as np
import pandas as pd

import torch

#from transvae import *
from transvae.transformer_models import TransVAE
from transvae.rnn_models import RNN, RNNAttn
from scripts.parsers import model_init, train_parser

def train(args):
    print("train function called /n")
    ### Update beta init parameter from loaded chekpoint
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=torch.device('cuda'))
        start_epoch = ckpt['epoch']
        total_epochs = start_epoch + args.epochs
        beta_init = (args.beta - args.beta_init) / total_epochs * start_epoch
        args.beta_init = beta_init

    if 'ON' in args.DDP: #bizare behaviour of arg parser with booleans means we convert here...
        args.DDP= True
    else:
        args.DDP=False
    if 'ON' in args.property_predictor:
        property_predictor= True
    else: 
        args.property_predictor = False
    ### Build params dict from the parsed arguments
    params = {'ADAM_LR': args.adam_lr,
              'ANNEAL_START': args.anneal_start,
              'BATCH_CHUNKS': args.batch_chunks,
              'BATCH_SIZE': args.batch_size,
              'BETA': args.beta,
              'BETA_INIT': args.beta_init,
              'EPS_SCALE': args.eps_scale,
              'HARDWARE' : args.hardware,
              'LR_SCALE': args.lr_scale,
              'WARMUP_STEPS': args.warmup_steps,
              'INIT_METHOD': args.init_method,
              'DIST_BACKEND': args.dist_backend,
              'WORLD_SIZE': args.world_size,
              'DISTRIBUTED': args.distributed,
              'NUM_WORKERS': args.num_workers,
              'DDP': args.DDP,
              'DISCRIMINATOR_LAYERS' : args.discriminator_layers}

    ### Load data, vocab and token weights
    train_mols = pd.read_csv('data/{}_train.txt'.format(args.data_source)).to_numpy()
    test_mols = pd.read_csv('data/{}_test.txt'.format(args.data_source)).to_numpy()
#     print('\n\n train shape: ',train_mols.shape, train_mols[0:5],'\n\n test shape', test_mols.shape, test_mols[0:5],'\n\n')#*************
    if args.property_predictor:
        assert args.train_props_path is not None and args.test_props_path is not None, \
        "ERROR: Must specify files with train/test properties if training a property predictor"
        train_props = pd.read_csv(args.train_props_path).to_numpy()
        test_props = pd.read_csv(args.test_props_path).to_numpy()
    else:
        train_props = None
        test_props = None
#     print("train_props",train_props.shape," test props shape: ",test_props.shape, '\n\n')
    with open('data/char_dict_{}.pkl'.format(args.data_source), 'rb') as f:
        char_dict = pickle.load(f)
    char_weights = np.load('data/char_weights_{}.npy'.format(args.data_source))
    params['CHAR_WEIGHTS'] = char_weights

    org_dict = {}
    for i, (k, v) in enumerate(char_dict.items()):
        if i == 0:
            pass
        else:
            org_dict[int(v-1)] = k

    params['CHAR_DICT'] = char_dict
    params['ORG_DICT'] = org_dict

    ### Train model
    vae = model_init(args, params)
    print(vae.model)
    if args.checkpoint is not None:
        vae.load(args.checkpoint)
    vae.train(train_mols, test_mols, train_props, test_props,
              epochs=args.epochs, save_freq=args.save_freq)


if __name__ == '__main__':
    print("main function called /n")
    parser = train_parser()
    args = parser.parse_args()
    train(args)
