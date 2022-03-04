import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image
from transvae import trans_models
from transvae.transformer_models import TransVAE
from transvae.rnn_models import RNN, RNNAttn
from transvae.wae_models import WAE
from transvae.aae_models import AAE
from transvae.tvae_util import *
from transvae import analysis
import glob
import re

from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.manifold import trustworthiness
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import plotly.express as px

import coranking #coranking.readthedocs.io
from coranking.metrics import trustworthiness, continuity, LCMC
from transvae.snc import SNC #github.com/hj-n/steadiness-cohesiveness



num_sequences = 45000
data_selection = "full_no_shuffle"

ckpt_list = glob.glob(""+"temp_ckpt//**//*.ckpt", recursive = True) #grab all checkpoint
loss_list = glob.glob(""+"temp_ckpt//**//*.txt", recursive = True) #grab all loss text files
print(os.getcwd(),ckpt_list)
for i in range(len(ckpt_list)):
    model_src = ckpt_list[i]
    src = loss_list[i]
    model_name = re.findall('([A-Z][a-zA-Z]{1,7})', model_src) #use regex to find model name in dir 
    print(model_name[0])
    model = locals()[model_name[0]](load_fn=model_src) #use locals to call model specific constructor
    
    latent_size = re.findall('(latent[\d]{2,3})', model_src)
    save_dir= "slurm_analyses//"+model.name+latent_size[0] #each model will have its own directory
    if not os.path.exists(save_dir):os.mkdir(save_dir) 
    save_dir= save_dir+"//" #actually enter the folder that was created above
    save_df = pd.DataFrame() #this will hold the number variables and save to CSV

    gpu = True

    if "full_no_shuffle" in data_selection:
        data = pd.read_csv('slurm_analyses//data//peptide_combined_no_shuff.txt').to_numpy() 
    elif "training" in data_selection:
        data = pd.read_csv('slurm_analyses//data//train_test//peptide_train.txt').to_numpy()
    elif "testing" in data_selection:
        data = pd.read_csv('slurm_analyses//data//train_test//peptide_test.txt').to_numpy()
    else:
        data = pd.read_csv('slurm_analyses//data//train_test//.txt').to_numpy() 
    data_1D = data[:num_sequences,0] #gets rid of extra dimension
    
    
    tot_loss = analysis.plot_loss_by_type(src,loss_types=['tot_loss'])
    plt.savefig(save_dir+'tot_loss.png')
    recon_loss = analysis.plot_loss_by_type(src,loss_types=['recon_loss'])
    plt.savefig(save_dir+'recon_loss.png')
    kld_loss = analysis.plot_loss_by_type(src,loss_types=['kld_loss'])
    plt.savefig(save_dir+'kld_loss.png')
    prob_bce_loss = analysis.plot_loss_by_type(src,loss_types=['prop_bce_loss'])
    plt.savefig(save_dir+'prob_bce_loss.png')
    if 'aae' in src:
        disc_loss = analysis.plot_loss_by_type(src,loss_types=['disc_loss'])
        plt.savefig(save_dir+'disc_loss.png')
    if 'wae' in src:
        mmd_loss = analysis.plot_loss_by_type(src,loss_types=['mmd_loss'])
        plt.savefig(save_dir+'mmd_loss.png')
    
    model.params['BATCH_SIZE'] = 200 #batch size must match total size of input data
    reconstructed_seq, props = model.reconstruct(data[:num_sequences], log=False, return_mems=False)
   
    if gpu:
        torch.cuda.empty_cache() #free allocated CUDA memory
    
    save_df['reconstructions'] = reconstructed_seq #placing the saves on a line separate from the ops allows for editing
    save_df['predicted properties'] = [prop.item() for prop in props[:len(reconstructed_seq)]]

    true_props_data = pd.read_csv('slurm_analyses//data//function_full_no_shuff.txt').to_numpy()
    true_props = true_props_data[0:num_sequences,0]
    prop_acc, prop_conf, MCC=calc_property_accuracies(props[:len(reconstructed_seq)],true_props[:len(reconstructed_seq)], MCC=True)
    
    save_df['property prediction accuracy'] = prop_acc
    save_df['property prediction confidence'] = prop_conf
    save_df['MCC'] = MCC
    
    
    # First we tokenize the input and reconstructed smiles
    input_sequences = []
    for seq in data_1D:
        input_sequences.append(peptide_tokenizer(seq))
    output_sequences = []
    for seq in reconstructed_seq:
        output_sequences.append(peptide_tokenizer(seq))
    
    seq_accs, tok_accs, pos_accs, seq_conf, tok_conf, pos_conf  = calc_reconstruction_accuracies(input_sequences, output_sequences)
    save_df['sequence accuracy'] = seq_accs
    save_df['sequence confidence'] = seq_conf
    save_df['token accuracy'] = tok_accs
    save_df['token confidence'] = tok_conf
    save_df = pd.concat([pd.DataFrame({'position_accs':pos_accs,'position_confidence':pos_conf }), save_df], axis=1)
    
    if model.model_type =='aae':
        mus, _, _ = model.calc_mems(data[:], log=False, save=False) 
    elif model.model_type == 'wae':
        mus, _, _ = model.calc_mems(data[:], log=False, save=False) 
    else:
        mems, mus, logvars = model.calc_mems(data[:], log=False, save=False) #subset size 1200*35=42000 would be ok
    
    
    vae_entropy_mus = calc_entropy(mus)
    save_df = pd.concat([save_df,pd.DataFrame({'mu_entropies':vae_entropy_mus})], axis=1)
    if model.model_type != 'wae' and model.model_type!= 'aae': #these don't have a variational type bottleneck
        vae_entropy_mems  = calc_entropy(mems)
        save_df = pd.concat([save_df,pd.DataFrame({'mem_entropies':vae_entropy_mems})], axis=1)
        vae_entropy_logvars = calc_entropy(logvars)
        save_df = pd.concat([save_df,pd.DataFrame({'logvar_entropies':vae_entropy_logvars})], axis=1)
    
   

    #create random index and re-index ordered memory list creating n random sub-lists (ideally resulting in IID random lists)
    random_idx = np.random.permutation(np.arange(stop=mus.shape[0]))
    mus[:] = mus[random_idx]
    data = data[random_idx]

    subsample_start=0
    subsample_length=mus.shape[0] #this may change depending on batch size

    #(for length based coloring): record all peptide lengths iterating through input
    pep_lengths = []
    for idx, pep in enumerate(data[subsample_start:(subsample_start+subsample_length)]):
        pep_lengths.append( len(pep[0]) )   
    #(for function based coloring): pull function from csv with peptide functions
    s_to_f =pd.read_csv("slurm_analyses//data//sequence_function_link_std_only.csv")    
    function = s_to_f["Antimicrobial"][subsample_start:(subsample_start+subsample_length)]
    function = function[random_idx] #account for random permutation

    pca = PCA(n_components=4)
    pca_batch =pca.fit_transform(X=mus[:])

    fig = px.scatter_matrix(pca_batch, color= pep_lengths, opacity=0.8)
    fig.write_image(save_dir+'pca_length.png', width=1200, height=800)

    fig = px.scatter_matrix(pca_batch, color= [str(itm) for itm in function], opacity=0.8)
    fig.write_image(save_dir+'pca_function.png', width=1200, height=800)

    #create n subsamples and calculate silhouette score for each
    latent_mem_func_subsamples = []
    pca_func_subsamples = []
    n=35
    for s in range(n):
        s_len = len(mus)//n #sample lengths
        mem_func_sil = metrics.silhouette_score(mus[s_len*s:s_len*(s+1)], function[s_len*s:s_len*(s+1)], metric='euclidean')
        latent_mem_func_subsamples.append(mem_func_sil)
        XY = [i for i in zip(pca_batch[s_len*s:s_len*(s+1),0], pca_batch[s_len*s:s_len*(s+1),1])]
        pca_func_sil = metrics.silhouette_score(XY, function[s_len*s:s_len*(s+1)], metric='euclidean')
        pca_func_subsamples.append(pca_func_sil)
    save_df = pd.concat([save_df,pd.DataFrame({'latent_mem_func_silhouette':latent_mem_func_subsamples})], axis=1)
    save_df = pd.concat([save_df,pd.DataFrame({'pca_func_silhouette':pca_func_subsamples})], axis=1)

    save_df.to_csv(save_dir+"saved_info.csv", index=False)
        