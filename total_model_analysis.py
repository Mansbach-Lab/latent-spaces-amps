import numpy as np
import random
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
from sklearn.manifold import Isomap
from sklearn import metrics
from sklearn.manifold import trustworthiness
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import plotly.express as px
import plotly.graph_objects as go
import Bio
from Bio import pairwise2
from Bio.Align import substitution_matrices

"""
This code is to be run from the "script for combined model analysis notebook"
Below we load the checkpoints from each model for all latent space sizes and run a series of benchmarks on them
The benchmarks are as follows:
    1) Plot the loss curves using the output files from training
    2) Reconstruction accuracy and sequence metrics
    3) PCA and latent space distribution metrics
    4) Sampling metrics and AMP sampling benchmarks

(the naming scheme of the checkpoint files is important in order to get this code to successively load all the checkpoints)
"""


def loss_plots(loss_src):
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
    plt.close('all')
    
def load_reconstructions(data,data_1D,latent_size, load_src, true_props=None,subset=None):
    
    recon_src = load_src+model.name+"_"+re.split('(\d{2,3})',latent_size[0])[0]+"_"+re.split('(\d{2,3})',latent_size[0])[1]+"//saved_info.csv"
    recon_df = pd.read_csv(recon_src)
    reconstructed_seq = recon_df['reconstructions'].to_list()[:num_sequences]
    props = torch.Tensor(recon_df['predicted properties'][:num_sequences])
    true_props_data = pd.read_csv(true_props).to_numpy()
    true_props = true_props_data[0:num_sequences,0]
    
    if subset:
        testing = pd.read_csv(subset).to_numpy()
        test_idx_list = [np.where(data==testing[idx][0]) for idx in range(len(testing))]


        batch_recon_len = len(reconstructed_seq)
        reconstructed_seq = [reconstructed_seq[test_idx_list[i][0][0]] for i in range(len(test_idx_list)) if test_idx_list[i][0][0]<batch_recon_len]
        data_1D= [data_1D[test_idx_list[i][0][0]] for i in range(len(test_idx_list)) if test_idx_list[i][0][0]<batch_recon_len]
        props = [props[test_idx_list[i][0][0]] for i in range(len(test_idx_list)) if test_idx_list[i][0][0]<batch_recon_len]
        props=torch.Tensor(props)
        data = testing[:][0]
        true_props_data = pd.read_csv(true_props).to_numpy()
        true_props = true_props_data[0:num_sequences,0]
        true_props= [true_props[test_idx_list[i][0][0]] for i in range(len(test_idx_list)) if test_idx_list[i][0][0]<batch_recon_len]

    return data, data_1D, true_props, props, reconstructed_seq

########################################################################################
gpu = True

num_sequences = 500_000
batch_size = 200 #setting for reconstruction
example_data = 'data/peptide_test.txt'
save_dir_loc = 'model_analyses\\test\\' #folder in which to save outpts
save_dir_name = 'test' #appended to identify data: train|test|other|etc...

reconstruct=True #True:reconstruct data here; False:load reconstructions from file
recon_src = "checkpointz/analyses_ckpts/" #directory in which all reconstructions are stored
true_prop_src = "data/function_test.txt" #if property predictor load the true labels
subset_src = "" #(optional) this file should have the true sequences for a subset of the "example data" above

phys_chem_props_dir = 'data//'

ckpt_list = glob.glob("checkpointz//all_300_ckpts//**//*.ckpt", recursive = True) #grab all checkpoint
print('current working directory: ',os.getcwd())



for i in range(len(ckpt_list)):
    
    #search the current directory for the model name and load that model
    model_dic = {'trans':'TransVAE','aae':'AAE','rnnattn':'RNNAttn','rnn':'RNN','wae':'WAE'}
    model_src = ckpt_list[i]
    print('working on: ',model_src,'\n')
    model_name = list(filter(None,[key for key in model_dic.keys() if key in model_src.split('\\')[-1]]))
    model = locals()[model_dic[model_name[0]]](load_fn=model_src) #use locals to call model specific constructor
    
    #create save directory for the current model according to latent space size
    latent_size = re.findall('(latent[\d]{2,3})', model_src)
    save_dir= save_dir_loc+model.name+"_"+latent_size[0]+"_"+save_dir_name
    if not os.path.exists(save_dir):os.mkdir(save_dir) 
    save_dir= save_dir+"//" 
    save_df = pd.DataFrame() #this will hold the number variables and save to CSV
    
    #load the true labels
    data = pd.read_csv(example_data).to_numpy() 
    data_1D = data[:num_sequences,0] #gets rid of extra dimension
    true_props_data = pd.read_csv(true_prop_src).to_numpy()
    true_props = true_props_data[0:num_sequences,0]

    # print("data loaded")
    # #get the log.txt file from the ckpt and model name then plot loss curves
    # loss_src = '_'.join( ("log",model_src.split('\\')[-1].split('_')[1],model_src.split('\\')[-1].split('_')[2][:-4]+"txt") )
    # src= '\\'.join([str(i) for i in model_src.split('\\')[:-1]])+"\\"+loss_src
    # print(loss_src, src)
    # loss_plots(src)
    
    # # #set the batch size and reconstruct the data (alternatively load the reconstructions from file)
    model.params['BATCH_SIZE'] = batch_size

    # if reconstruct:
    #     reconstructed_seq, props = model.reconstruct(data[:num_sequences], log=False, return_mems=False)
    # else:
    #     data, data_1D, true_props, props, reconstructed_seq = load_reconstructions(data, data_1D,latent_size,
    #                                                                                load_src=recon_src,
    #                                                                                true_props=true_prop_src)
    # if gpu:torch.cuda.empty_cache() #free allocated CUDA memory
#     #save the metrics to the dataframe
#     save_df['reconstructions'] = reconstructed_seq #placing the saves on a line separate from the ops allows for editing
#     save_df['predicted properties'] = [prop.item() for prop in props[:len(reconstructed_seq)]]
#     prop_acc, prop_conf, MCC=calc_property_accuracies(props[:len(reconstructed_seq)],true_props[:len(reconstructed_seq)], MCC=True)
#     save_df['property prediction accuracy'] = prop_acc
#     save_df['property prediction confidence'] = prop_conf
#     save_df['MCC'] = MCC
# #   First we tokenize the input and reconstructed smiles
#     input_sequences = []
#     for seq in data_1D:
#         input_sequences.append(peptide_tokenizer(seq))
#     output_sequences = []
#     for seq in reconstructed_seq:
#         output_sequences.append(peptide_tokenizer(seq))
#     seq_accs, tok_accs, pos_accs, seq_conf, tok_conf, pos_conf = calc_reconstruction_accuracies(input_sequences, output_sequences)
#     save_df['sequence accuracy'] = seq_accs
#     save_df['sequence confidence'] = seq_conf
#     save_df['token accuracy'] = tok_accs
#     save_df['token confidence'] = tok_conf
#     save_df = pd.concat([pd.DataFrame({'position_accs':pos_accs,'position_confidence':pos_conf }), save_df], axis=1)
    
    ##moving into memory and entropy
    if model.model_type =='aae':
        mus, _, _ = model.calc_mems(data[:61_000], log=False, save=False) #50_000
    elif model.model_type == 'wae':
        mus, _, _ = model.calc_mems(data[:61_000], log=False, save=False) 
    else:
        mems, mus, logvars = model.calc_mems(data[:61_000], log=False, save=False) #subset size 1200*35=42000 would be ok


    # # ##calculate the entropies (DEPRECATED<--)
    # # vae_entropy_mus = calc_entropy(mus)
    # # save_df = pd.concat([save_df,pd.DataFrame({'mu_entropies':vae_entropy_mus})], axis=1)
    # # if model.model_type != 'wae' and model.model_type!= 'aae': #these don't have a variational type bottleneck
    # #     vae_entropy_mems  = calc_entropy(mems)
    # #     save_df = pd.concat([save_df,pd.DataFrame({'mem_entropies':vae_entropy_mems})], axis=1)
    # #     vae_entropy_logvars = calc_entropy(logvars)
    # #     save_df = pd.concat([save_df,pd.DataFrame({'logvar_entropies':vae_entropy_logvars})], axis=1)

    # #create random index and re-index ordered memory list
    # #random_idx = np.random.permutation(np.arange(start=0, stop=mus.shape[0]))
    # #mus = mus[random_idx]
    # shuf_data = data[random_idx]

    shuf_data=data #not shuffling anymore 
    subsample_start=0
    subsample_length=mus.shape[0] #mus shape depends on batch size!

    #(for length based coloring): record all peptide lengths iterating through input
    pep_lengths = []
    for idx, pep in enumerate(shuf_data[subsample_start:(subsample_start+subsample_length)]):
        pep_lengths.append( len(pep[0]) )   
    #(for function based coloring): pull function from csv with peptide functions
    s_to_f =pd.read_csv(true_prop_src)    
    function = s_to_f['peptides'][subsample_start:(subsample_start+subsample_length)]
    #function = function[random_idx] #account for random permutation

    pca = PCA(n_components=5,svd_solver='full')
    pca_batch =pca.fit_transform(X=mus[:])

    # #Calculate and plot the loading matrix from the PCA fit of the data
    # loadings = pca.components_.T*np.sqrt(pca.explained_variance_)
    # color=['tab:blue','tab:red','tab:green','tab:orange','tab:purple']
    # y_labels=['PC1 Correlation','PC2 Correlation','PC3 Correlation','PC4 Correlation','PC5 Correlation']
    # titles=['Latent Dimension Correlations to PC1','Latent Dimension Correlations to PC2',
    #         'Latent Dimension Correlations to PC3','Latent Dimension Correlations to PC4',
    #         'Latent Dimension Correlations to PC5']
    # for pc in range (loadings.shape[1]):
    #     plt.figure(figsize=(10,6))
    #     plt.title(titles[pc])
    #     plt.ylabel(y_labels[pc])
    #     plt.xlim(-1,loadings.shape[0]+1)
    #     plt.xlabel('Latent Dimensions')
    #     plt.bar(np.linspace(0,loadings.shape[0]-1,loadings.shape[0]),loadings[:,pc])
    #     plt.savefig(save_dir+'latent_correlations_PC{}.png'.format(pc+1),facecolor='white',transparent=None,bbox_inches='tight',dpi=600)
    #     plt.close()

    # #plot format dictionnaries
    titles={'text':'{}'.format(model.model_type.replace("_"," ").upper()),
                          'x':0.5,'xanchor':'center','yanchor':'top','font_size':40}
    general_fonts={'family':"Helvetica",'size':30,'color':"Black"}
    colorbar_fmt={'title_font_size':30,'thickness':15,'ticks':'','title_text':'Lengths',
                               'ticklabelposition':"outside bottom"}
    
    # fig = px.scatter(pd.DataFrame({"PC1":pca_batch[:,0],"PC2":pca_batch[:,1], "lengths":pep_lengths}),
    #             symbol_sequence=['hexagon2'],x='PC1', y='PC2', color="lengths",
    #             color_continuous_scale='Jet',template='simple_white', opacity=0.9)
    # fig.update_traces(marker=dict(size=9))
    # fig.update_layout(title=titles,xaxis_title="PC1", yaxis_title="PC2",font=general_fonts)
    # fig.update_coloraxes(colorbar=colorbar_fmt)
    # fig.write_image(save_dir+'pca_length.png', width=900, height=600)

    pc_pairs = [[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
    # for pc_pair in pc_pairs:
    #     fig = px.scatter(pd.DataFrame({"PC1":pca_batch[:,pc_pair[0]],"PC2":pca_batch[:,pc_pair[1]],
    #                                     #"Function":list(function.map({0:"NON-AMP", 1:"UNI_AMP", 2:"STAR_AMP"}))}),
    #                                     "Function":list(map(lambda itm: "AMP" if itm==1 else "NON-AMP",function))}),
    #                                     x='PC1', y='PC2', color="Function",symbol_sequence=['x-thin-open','circle', 'circle'],
    #                                     template='simple_white',symbol='Function', opacity=0.8) 
    #     fig.update_traces(marker=dict(size=9))
    #     fig.update_layout(title=titles,xaxis_title="PC{}".format(pc_pair[0]),yaxis_title="PC{}".format(pc_pair[1]),font=general_fonts)
    #     fig.write_image(save_dir+'pca[{},{}]_function.png'.format(pc_pair[0],pc_pair[1]), width=2_000, height=1_000)
    
    # Plot the explained variances
    # plt.figure()
    # plt.bar(range(pca.n_components_), pca.explained_variance_ratio_*100, color='black')
    # plt.xlabel('PCA features')
    # plt.ylabel('variance %')
    # plt.xticks(range(pca.n_components_))
    # plt.savefig(save_dir+model.name+latent_size[0]+'.png', facecolor='white',transparent=None,)
    # plt.close()

    # Section to plot all PCs simultaneously DEPRECATED <--
    # fig = px.scatter_matrix(pd.DataFrame({"PC1":pca_batch[:,0],"PC2":pca_batch[:,1],"PC3":pca_batch[:,2],
    #                                       "PC4":pca_batch[:,3],"PC5":pca_batch[:,4],"lengths":pep_lengths}),
    #                                 dimensions=["PC1","PC2","PC3","PC4","PC5"],
    #                                 symbol_sequence=['hexagon2'],template='simple_white',
    #                                 color="lengths",color_continuous_scale='Jet', opacity=0.9)
    # fig.update_traces(diagonal_visible=False)
    # fig.update_layout(title=titles,font=general_fonts)
    # #fig.write_image(save_dir+'pca_matrix_length.png', width=5_000, height=2500)
    # fig.write_image(save_dir+'pca_matrix_length.png', width=3_000, height=1_500) 
    
    # fig = px.scatter_matrix(pd.DataFrame({"PC1":pca_batch[:,0],"PC2":pca_batch[:,1],"PC3":pca_batch[:,2],
    #                                       "PC4":pca_batch[:,3],"PC5":pca_batch[:,4],
    #                                "Function":list(map(lambda itm: "AMP" if itm==1 else "NON-AMP",function))}),
    #                                 dimensions=["PC1","PC2","PC3","PC4","PC5"],template='simple_white',
    #                                 color="Function",symbol_sequence=['x-thin','circle'],
    #                                 symbol='Function', opacity=0.8) 
    # fig.update_traces(diagonal_visible=False)
    # fig.update_layout(title=titles,font=general_fonts)
    # #fig.write_image(save_dir+'pca_matrix_function.png', width=5_000, height=2500)
    # fig.write_image(save_dir+'pca_matrix_function.png', width=3_000, height=1_500) 

    # pearson = {} #dict to store the pearson coefficient between PCA vs AMP function or physicochem.props.
    # if 'train' in save_dir_name:
    #     phys_props = pd.read_csv(phys_chem_props_dir+'train_physicochem_props.csv')
    # else:
    #     phys_props = pd.read_csv(phys_chem_props_dir+'test_physicochem_props.csv')

    # first calculate silhouette score on all latent space dims
    n=15
    latent_mem_func_subsamples = []
    for s in range(n):
        s_len = len(mus)//n #sample lengths
        mem_func_sil = metrics.silhouette_score(mus[s_len*s:s_len*(s+1)], function[s_len*s:s_len*(s+1)], metric='euclidean')
        latent_mem_func_subsamples.append(mem_func_sil)
    save_df = pd.concat([save_df,pd.DataFrame({'latent_mem_func_silhouette':latent_mem_func_subsamples})], axis=1)
    top_5_single_pc_silhouettes=[]
    #then go over single & pairs of PC's from PCA and find max SS PC's (subsampling for time complexity)
    for idx, pc_pair in enumerate(pc_pairs):
        print("working on PC[{},{}]".format(pc_pair[0],pc_pair[1]))
        pca_func_subsamples = []
        single_pc_silhouette_score=[]
        s_len = len(mus)//n #sample lengths
        for s in range(n): #go through subsamples
            if idx<5:#only use the first 5 iterations to sample the single PCs for silhouette score
                single_pc_silhouette_score.append(metrics.silhouette_score(pca_batch[s_len*s:s_len*(s+1),idx].reshape(-1,1),function[s_len*s:s_len*(s+1)]))
            XY = [i for i in zip(pca_batch[s_len*s:s_len*(s+1),pc_pair[0]], pca_batch[s_len*s:s_len*(s+1),pc_pair[1]])]
            pca_func_subsamples.append(metrics.silhouette_score(XY, function[s_len*s:s_len*(s+1)], metric='euclidean'))
        if idx<5:
            top_5_single_pc_silhouettes.append(np.average(single_pc_silhouette_score))
        save_df = pd.concat([save_df,pd.DataFrame({'pca_func_silhouette[{},{}]'.format(pc_pair[0],pc_pair[1]):pca_func_subsamples})], axis=1)
    
    # pearson.update({'amp_silhouette':top_5_single_pc_silhouettes})
    # save_df.to_csv(save_dir+"saved_info.csv", index=False)
    
    # #calculate the remaining correlations on physicochemical properties
    # for col in phys_props.columns:
    #     functions = phys_props[col][:len(mus)].values
    #     pearson.update({str(col)+'_pearsonr':[(pearsonr(pca_batch[:,pc],functions)[0],pearsonr(pca_batch[:,pc],functions)[1]) for pc in range(5)]})
    #     fig = px.scatter_matrix(pd.DataFrame({"PC1":pca_batch[:,0],"PC2":pca_batch[:,1],"PC3":pca_batch[:,2],
    #                                             "PC4":pca_batch[:,3],"PC5":pca_batch[:,4],
    #                                     "Function":functions}),
    #                                         dimensions=["PC1","PC2","PC3","PC4","PC5"],template='simple_white',
    #                                         color="Function",opacity=0.9) 
    #     colorbar_fmt={'title_font_size':30,'thickness':15,'ticks':'','title_text':str(col),
    #                         'ticklabelposition':"outside bottom"}
    #     fig.update_traces(diagonal_visible=False)
    #     fig.update_layout(title=titles,font=general_fonts)
    #     fig.update_coloraxes(colorbar=colorbar_fmt, 
    #                         cmax=np.mean(functions)+np.std(functions),
    #                         cmin=np.mean(functions)-np.std(functions),
    #                         cmid=np.mean(functions))
    #     #fig.write_image(save_dir+col+'_PCA_matrix'+'.png', width=5_000, height=2500)
    #     fig.write_image(save_dir+col+'_PCA_matrix'+'.png', width=2_000, height=1_000)
    # df_pearson = pd.DataFrame.from_dict(pearson)
    # df_pearson.to_csv(save_dir+'pearsonr.csv', index=False)

    #Section dealing with sequence generation metrics and bootstrapping from the latent space
    #first randomly sample points within the latents space
    # rnd_seq_count =1_000 
    # rnd_latent_list=[] #generate N latent space vectors

    # mu_avg = np.average(mus,axis=0)
    # mu_var = np.var(mus, axis=0)

    # for seq in range(rnd_seq_count):
    #     rnd_latent_list.append(np.random.normal(mu_avg, mu_var, size=(mus.shape[1])).astype(np.float32))
    
    # model.params['BATCH_SIZE'] = 25
    # rnd_token_list=np.empty((rnd_seq_count,model.tgt_len)) #store N decoded latent vectors now in token(0-20) form max length 125
    
    # #decode these points into predicted amino acid tokens (integers)
    # for batch in range(0,rnd_seq_count,model.params['BATCH_SIZE']):
    #     rnd_token_list[batch:batch+model.params['BATCH_SIZE']] =  model.greedy_decode(torch.tensor(rnd_latent_list[batch:batch+model.params['BATCH_SIZE']]).cuda()).cpu()
    
    # #turn the tokens into characters
    # decoded_rnd_seqs = decode_mols(torch.tensor(rnd_token_list), model.params['ORG_DICT'])
    # decoded_rnd_seqs[:]=[x for x in decoded_rnd_seqs if x] #removes the empty lists
    
    # df_gen_scores = {} #dictionnary to store results
    # #UNIQUENESS
    # percent_unique, unique_conf = uniqueness(decoded_rnd_seqs)
    # df_gen_scores.update({'percent_unique': percent_unique})
    # df_gen_scores.update({'unique_confidence':unique_conf})
    
    # #NOVELTY
    # #sample N test/train set sequences randomly and compare to those created
    # percent_novel, novel_conf = novelty(data, np.expand_dims(np.array(decoded_rnd_seqs),1))
    # df_gen_scores.update({'percent_novel':percent_novel})
    # df_gen_scores.update({'novel_confidence':novel_conf})
    
    # #AMP SAMPLING
    # peptides_to_probe=10
    # sample_count=100
    best_pc = np.argmax(top_5_single_pc_silhouettes) #find the best PCvsAMP correlation
    # pca_mean = np.average(pca_batch[:,best_pc])
    # pca_std = np.std(pca_batch[:,best_pc])
    # pca_scan = np.zeros((peptides_to_probe,5)) #create a reduced vector to be sent backwards to high-D
    # pca_scan[:,best_pc]=np.linspace(start=pca_mean-(4*pca_std), stop=pca_mean+(4*pca_std), num=peptides_to_probe) #scan 1 dim evenly with best PC
    # amp_sample_latents = pca.inverse_transform(pca_scan) #inverse to high-Dims for decoding
    # all_gen_seqs = [] #stored in a text file for AMP prediction later
    # for idx,amp in enumerate(amp_sample_latents):
    #     print("working on amp sample number: ",idx)
    #     nearby_samples = np.random.normal(loc=amp,scale=pca_std/5,size=(sample_count,1,model.params['d_latent'])).astype(np.float32)
    #     model.params['BATCH_SIZE'] = 25
    #     rnd_token_list=np.empty((sample_count,model.tgt_len)) #store N decoded latent vectors now in token(0-20) form max length 125
    #     for batch in range(0,sample_count,model.params['BATCH_SIZE']):
    #         rnd_token_list[batch:batch+model.params['BATCH_SIZE']] =  model.greedy_decode(torch.tensor(nearby_samples[batch:batch+model.params['BATCH_SIZE']]).squeeze().cuda()).cpu()
    #     decoded_rnd_seqs = decode_mols(torch.tensor(rnd_token_list), model.params['ORG_DICT'])
        
    #     for seq in decoded_rnd_seqs:
    #         if len(seq)<=50 and len(seq)>0: #save only sequences with length <=50
    #             all_gen_seqs.append(seq) #appending to list of all generated sequences
    #     decoded_rnd_seqs = [seq for seq in decoded_rnd_seqs if len(seq)>0 and len(seq)<=50] #keep constrained length seqs
        
    #     print("starting Sequence similarity")
    #     #SEQ SIMILARITY
    #     #(n^2+n)/2 time complexity on sequence similarity 100 seqs=7 mins and 50 seqs=30seconds so split into 5 groups of 20
    #     div=5
    #     similarity_score = np.array([sequence_similarity(decoded_rnd_seqs[i*sample_count//div:(i+1)*sample_count//div]) for i in range(div)])
    #     df_gen_scores.update({'average_sequence_similarity_'+str(idx): np.average(similarity_score)})
    #     df_gen_scores.update({'std_on_similarity_score_'+str(idx): np.std(similarity_score)})
        
    #     #AMP UNIQUENESS
    #     amp_percent_unique, amp_unique_conf = uniqueness(decoded_rnd_seqs)
    #     df_gen_scores.update({'amp_uniqueness_'+str(idx): amp_percent_unique})
    #     df_gen_scores.update({'amp_uniqueness_std_'+str(idx): amp_unique_conf})
    
    #     #Jaccard Similarity Score
    #     jac_scores_2 = jaccard_similarity_score(decoded_rnd_seqs,2)
    #     jac_scores_3 = jaccard_similarity_score(decoded_rnd_seqs,3)
    #     df_gen_scores.update({'amp_jac_score_2_'+str(idx): np.average(jac_scores_2)})
    #     df_gen_scores.update({'amp_jac_score_std_2_'+str(idx): np.std(jac_scores_2)})
    #     df_gen_scores.update({'amp_jac_score_3_'+str(idx): np.average(jac_scores_3)})
    #     df_gen_scores.update({'amp_jac_score_std_3_'+str(idx): np.std(jac_scores_3)})
    
    # #Store Output
    # with open(save_dir+'all_gen_seqs.txt','w') as f:
    #     for seq in all_gen_seqs:
    #         f.write(str(seq)+"\n")
    # f.close()
    with open(save_dir+'PC_minmax.txt','w') as f:
        pca_min = np.min(pca_batch[:,best_pc])
        pca_max = np.max(pca_batch[:,best_pc])
        f.write(str(pca_min))
        f.write('\t')
        f.write(str(pca_max))
    f.close()
    # with open(save_dir+'PC_meanstd.txt','w') as f:
    #     f.write(str(pca_mean))
    #     f.write('\t')
    #     f.write(str(pca_std))
    # f.close()
    # df = pd.DataFrame.from_dict([df_gen_scores])
    # pd.DataFrame.from_dict([df_gen_scores]).to_csv(save_dir+"generation_metrics.csv", index=False)
    