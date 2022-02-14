import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image
#from transvae import trans_models
#from transvae.transformer_models import TransVAE
#from transvae.rnn_models import RNN, RNNAttn
#from transvae.wae_models import WAE
#from transvae.aae_models import AAE
#from transvae.tvae_util import *
#from transvae import analysis
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
        data = pd.read_csv('notebooks//example_data//peptide_combined_no_shuff.txt').to_numpy() 
    elif "training" in data_selection:
        data = pd.read_csv('notebooks//example_data//train_test//peptide_train.txt').to_numpy()
    elif "testing" in data_selection:
        data = pd.read_csv('notebooks//example_data//train_test//peptide_test.txt').to_numpy()
    else:
        data = pd.read_csv('notebooks//example_data//train_test//.txt').to_numpy() 
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

    true_props_data = pd.read_csv('notebooks//example_data//function_full_no_shuff.txt').to_numpy()
    true_props = true_props_data[0:num_sequences,0]
    prop_acc ,MCC = calc_property_accuracies(props[:len(reconstructed_seq)],true_props[:len(reconstructed_seq)], MCC=True)
    
    save_df['property prediction accuracy'] = prop_acc
    save_df['MCC'] = MCC
    
    
    # First we tokenize the input and reconstructed smiles
    input_sequences = []
    for seq in data_1D:
        input_sequences.append(peptide_tokenizer(seq))
    output_sequences = []
    for seq in reconstructed_seq:
        output_sequences.append(peptide_tokenizer(seq))
    
    seq_accs, token_accs, position_accs = calc_reconstruction_accuracies(input_sequences, output_sequences)
    save_df['sequence accuracy'] = seq_accs
    save_df['token accuracy'] = token_accs
    save_df = pd.concat([save_df, pd.DataFrame({'position_accs':position_accs})], axis=1)
    
    
    if model.model_type =='aae':
        mems, _, _ = model.calc_mems(data[:8000], log=False, save=False) 
    elif model.model_type == 'wae':
        mems, _, _ = model.calc_mems(data[:8000], log=False, save=False) 
    else:
        mems, mus, logvars = model.calc_mems(data[:8000], log=False, save=False) 
    
    
    vae_entropy_mems  = calc_entropy(mems)
    if model.model_type != 'wae' and model.model_type!= 'aae': #these don't have a variational type bottleneck
        vae_entropy_mus = calc_entropy(mus)
        vae_entropy_logvars = calc_entropy(logvars)

    total_entropy_mems = np.sum(vae_entropy_mems)
    print('The model memories contain {} nats of information'.format(round(total_entropy_mems, 2)))
    save_df['mem entropy']= total_entropy_mems
    if model.model_type != 'wae' and model.model_type!= 'aae':
        total_entropy_mus = np.sum(vae_entropy_mus)
        print('The model means contain {} nats of information'.format(round(total_entropy_mus, 2)))
        total_entropy_logvars = np.nansum(vae_entropy_logvars)
        print('The model logvar contain {} nats of information'.format(round(total_entropy_logvars, 2)))
        save_df['mu entropy']= total_entropy_mus
        save_df['logvar entropy']= total_entropy_logvars
    else:
        save_df['mu entropy']= 0
        save_df['logvar entropy']= 0
    
    #raw input processing
    flattened_data = data[:,0]
    max_len = 126
    char_dict = model.params['CHAR_DICT']
    input_sequences = []
    for seq in flattened_data:
        tokenized = (peptide_tokenizer(seq))
        input_sequences.append(encode_seq(tokenized, max_len, char_dict))
    raw_input=input_sequences

    subsample_start=0
    subsample_length=mems.shape[0] #this may change depending on batch size

    pca = PCA(n_components=4)

    #only need to perform PCA once then can color code based on either length or function
    pca_batch =pca.fit_transform(X=mems[:])

    #(for length based coloring): record all peptide lengths iterating through input
    pep_lengths = []
    for idx, pep in enumerate(data[subsample_start:(subsample_start+subsample_length)]):
        pep_lengths.append( len(pep[0]) )
    fig = px.scatter_matrix(pca_batch, color= pep_lengths, opacity=0.8)
    fig.write_image(save_dir+'pca_length.png', width=1200, height=800)

    #(for function based coloring): pull function from csv with peptide functions
    s_to_f =pd.read_csv("data//peptides//amp_function//sequence_function_link_std_only.csv")    
    function = s_to_f["Antimicrobial"][subsample_start:(subsample_start+subsample_length)]
    fig = px.scatter_matrix(pca_batch, color= [str(itm) for itm in function], opacity=0.8)
    fig.write_image(save_dir+'pca_function.png', width=1200, height=800)

    mem_func_sil = metrics.silhouette_score(mems, function, metric='euclidean')
    print('Silhoutte score of latent memory & function: ',mem_func_sil)
    save_df['latent_to_func_sil'] = mem_func_sil


    XY = [i for i in zip(pca_batch[:,0], pca_batch[:,1])]
    pca_func_sil = metrics.silhouette_score(XY, function, metric='euclidean')
    print('Silhoutte score of PCA latent memory & function: ',pca_func_sil)
    save_df['PCA_to_func_sil']= pca_func_sil
    

    Q = coranking.coranking_matrix(mems, pca_batch)
    trust_pca = trustworthiness(Q, min_k=1, max_k=50)
    cont_pca = continuity(Q, min_k=1, max_k=50)
    lcmc_pca = LCMC(Q, min_k=1, max_k=50)
    print(np.mean(trust_pca),np.mean(cont_pca),np.mean(lcmc_pca))
    save_df['latent_to_PCA_trustworthiness'] = np.mean(trust_pca)
    save_df['latent_to_PCA_continuity'] = np.mean(cont_pca)
    save_df['latent_to_PCA_lcmc'] = np.mean(lcmc_pca)
    Q=0 #trying to free RAM 
    trust_pca=0
    cont_pca=0
    lcmc_pca=0
    torch.cuda.empty_cache() #free allocated CUDA memory
    #metrics for steadiness and cohisiveness from latent spcace to PCA
    # k value for computing Shared Nearest Neighbor-based dissimilarity 
    parameter = { "k": 50,"alpha": 0.1 }
    metrics = SNC(raw=mems, emb=pca_batch, iteration=300, dist_parameter = parameter)
    metrics.fit()
    save_df['latent_to_PCA_steadiness'] = metrics.steadiness()
    save_df['latent_to_PCA_cohesiveness'] =metrics.cohesiveness()
    print(metrics.steadiness(),metrics.cohesiveness())

    save_df.to_csv(save_dir+"saved_info.csv", index=False)
        