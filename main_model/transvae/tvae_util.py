import re
import math
import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import precision_score, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#from rdkit import rdBase
#from rdkit import Chem
#from rdkit.Chem import Descriptors
#from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan

#rdBase.DisableLog('rdApp.*')


######## MODEL HELPERS ##########

def clones(module, N):
    """Produce N identical layers (adapted from
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    """Mask out subsequent positions (adapted from
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)"""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention' (adapted from Viswani et al.)"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class ListModule(nn.Module):
    """Create single pytorch module from list of modules"""
    def __init__(self, *args):
        super().__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

class KLAnnealer:
    """
    Scales KL weight (beta) linearly according to the number of epochs
    """
    def __init__(self, kl_low, kl_high, n_epochs, start_epoch):
        self.kl_low = kl_low
        self.kl_high = kl_high
        self.n_epochs = n_epochs
        self.start_epoch = start_epoch

        self.kl = (self.kl_high - self.kl_low) / (self.n_epochs - self.start_epoch)

    def __call__(self, epoch):
        if self.start_epoch == 0:
            k = (epoch - self.start_epoch) if epoch >= self.start_epoch else 0
            beta = self.kl_low + k * self.kl
            if beta > self.kl_high:
                beta = self.kl_high  
            else:
                pass
        else: #when checkpointing just set the beta to the max value from previous training
            beta = self.kl_high
        return beta


####### PREPROCESSING HELPERS ##########

def tokenizer(smile):
    "Tokenizes SMILES string"
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(smile)]
    assert smile == ''.join(tokens), ("{} could not be joined".format(smile))
    return tokens

#inspired by the Trans-Vae tokenizer
def peptide_tokenizer(peptide):
    "Tokenizes SMILES string"
    #need to remove "X", "B", "Z", "U", "O"
    pattern =  "(G|A|L|M|F|W|K|Q|E|S|P|V|I|C|Y|H|R|N|D|T|X|B|Z|U|O)"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(peptide)]
    assert peptide == ''.join(tokens), ("{} could not be joined".format(peptide))
    return tokens

def build_org_dict(char_dict):
    org_dict = {}
    for i, (k, v) in enumerate(char_dict.items()):
        if i == 0:
            pass
        else:
            org_dict[int(v-1)] = k
    return org_dict

def encode_seq(sequence, max_len, char_dict):
    "Converts tokenized sequences to list of token ids"
    for i in range(max_len - len(sequence)):
        if i == 0:
            sequence.append('<end>')
        else:
            sequence.append('_')
    seq_vec = [char_dict[c] for c in sequence] #map all characters to their respective numbers
    return seq_vec

def get_char_weights(train_smiles, params, freq_penalty=0.5):
    "Calculates token weights for a set of input data"
    char_dist = {}
    char_counts = np.zeros((params['NUM_CHAR'],))
    char_weights = np.zeros((params['NUM_CHAR'],))
    for k in params['CHAR_DICT'].keys():
        char_dist[k] = 0
    for smile in train_smiles:
        for i, char in enumerate(smile):
            char_dist[char] += 1
        for j in range(i, params['MAX_LENGTH']):
            char_dist['_'] += 1
    for i, v in enumerate(char_dist.values()):
        char_counts[i] = v
    print(char_dist)
    top = np.sum(np.log(char_counts))
    for i in range(char_counts.shape[0]):
        char_weights[i] = top / np.log(char_counts[i])
    min_weight = char_weights.min()
    for i, w in enumerate(char_weights):
        if w > 2*min_weight:
            char_weights[i] = 2*min_weight
    scaler = MinMaxScaler([freq_penalty,1.0])
    char_weights = scaler.fit_transform(char_weights.reshape(-1, 1))
    return char_weights[:,0]


####### POSTPROCESSING HELPERS ##########

def decode_mols(encoded_tensors, org_dict):
    "Decodes tensor containing token ids into string"
    mols = []
    for i in range(encoded_tensors.shape[0]):
        encoded_tensor = encoded_tensors.cpu().numpy()[i,:] - 1
        mol_string = ''
        for i in range(encoded_tensor.shape[0]):
            idx = encoded_tensor[i]
            if org_dict[idx] == '<end>':
                break
            elif org_dict[idx] == '_':
                pass
            else:
                mol_string += org_dict[idx]
        mols.append(mol_string)
    return mols

def calc_reconstruction_accuracies(input_sequences, output_sequences):
    "Calculates sequence, token and positional accuracies for a set of\
    input and reconstructed sequences"
    max_len = 126
    seq_accs = []
    hits = 0 #used by token acc only
    misses = 0 #used by token acc only
    position_accs = np.zeros((2, max_len)) #used by pos acc only
    for in_seq, out_seq in zip(input_sequences, output_sequences):
        if in_seq == out_seq:
            seq_accs.append(1)
        else:
            seq_accs.append(0)

        misses += abs(len(in_seq) - len(out_seq)) #number of missed tokens in the prediction seq
        for j, (token_in, token_out) in enumerate(zip(in_seq, out_seq)): #look at individual tokens for current seq
            if token_in == token_out:
                hits += 1
                position_accs[0,j] += 1
            else:
                misses += 1
            position_accs[1,j] += 1

    seq_acc = np.mean(seq_accs) #list of 1's and 0's for correct or incorrect complete seq predictions
    token_acc = hits / (hits + misses)
    position_acc = []
    position_conf = []
    #calculating the confidence interval of the accuracy results
    z=1.96 #95% confidence interval
    for i in range(max_len):
        position_acc.append(position_accs[0,i] / position_accs[1,i])
        position_conf.append(z*math.sqrt(position_acc[i]*(1-position_acc[i])/position_accs[1,i]))
    
    seq_conf = z*math.sqrt(seq_acc*(1-seq_acc)/len(seq_accs))
    token_conf = z*math.sqrt(token_acc*(1-token_acc)/(hits+misses))
    
    return seq_acc, token_acc, position_acc, seq_conf, token_conf, position_conf

def calc_property_accuracies(pred_props, true_props, MCC=False):
    binary_predictions = torch.round(pred_props) #round the output float from the network to either 0 or 1
    TN = 0
    TP = 0
    FP = 0
    FN = 0
    for idx, prop in enumerate(binary_predictions):
        if true_props[idx] == 0 and prop == true_props[idx]:
            TN += 1
        if true_props[idx] == 1 and prop == true_props[idx]:
            TP += 1
        if true_props[idx] == 0 and prop != true_props[idx]:
            FP += 1
        if true_props[idx] == 1 and prop != true_props[idx]:
            FN += 1

    acc = (TN + TP) / (TN+TP+FP+FN)
    
    z=1.96 #95% confidence interval
    conf = z*math.sqrt(acc*(1-acc)/(TN+TP+FP+FN))
    
    print("property accuracy :",(TN + TP),"/",(TN+TP+FP+FN),"=",(TN + TP) / (TN+TP+FP+FN) )
    if MCC:
        N = TN + TP + FN + FP
        S = (TP + FN) / N
        P = (TP + FP) / N
        if S!=1 and P!=1:
            MCC = ( (TP/N)-(S*P) )/ math.sqrt(P*S*(1-S)*(1-P))
        else: MCC="Data error Division by zero"
        print("MCC: ", MCC)
    return acc, conf, MCC
   

def calc_entropy(sample):
    "Calculates Shannon information entropy for a set of input memories"
    es = []
    for i in range(sample.shape[1]):
        probs, bins = np.histogram(sample[:,i], bins=1000, range=(-5, 5), density=True)
        cur_ent = entropy(probs)
        es.append(cur_ent)
    return np.array(es)

####### PEPTIDE METRICS #########
# from sourmash.readthedocs.io/en/latest/kmers-and-minhash.html
from sklearn.metrics import jaccard_score
def build_kmers(sequence, ksize):
    kmers = []
    n_kmers = len(sequence) - ksize + 1

    for i in range(n_kmers):
        kmer = sequence[i:i + ksize]
        kmers.append(kmer)
        
    return kmers

def jaccard_similarity(a, b):
    n_a = set(a)
    n_b = set(b)

    intersection = len(n_a.intersection(n_b))
    union = len(n_a.union(n_b))
    return intersection / union

def jaccard_similarity_score(seq_list,k=2):
    import itertools
    set_1 = list(set(seq_list))
    combinations = list(itertools.combinations(seq_list,2))
    jac_scores = np.empty(len(combinations))
    for idx, combination in enumerate(combinations):
        if len(combination[0])<=k or len(combination[1])<=k:
            jac_scores[idx]=0
        else:
            jac_scores[idx] = jaccard_similarity(build_kmers(combination[0],k), build_kmers(combination[1],k))
    return jac_scores

def uniqueness(seq_list):
    "Returns the % of unique items in a list"
    z=1.96 #95% confidence interval
    percent_unique = len(set(seq_list)) / len(seq_list)
    unique_conf = z*math.sqrt(percent_unique*(1-percent_unique)/len(seq_list))
    return percent_unique, unique_conf
    
def novelty(new_sequences, dataset_sequences):
    """
    This function compares two numpy lists of sequences and returns the % of seqs that are novel in the new list
    new_sequences: newly generated list of sequences
    dataset_sequences: list of sequences to compare against
    """
    z=1.96 #95% confidence interval
    combined = np.concatenate((new_sequences, dataset_sequences)) #first combine both lists
    set_combined = set(combined.flatten().tolist()) #remove redundant seqs with "set"
    percent_novel = (len(set_combined)-len(dataset_sequences)) / (len(new_sequences)) #subtract data seqs and get %
    novel_conf =  z*math.sqrt(percent_novel*(1-percent_novel)/(len(new_sequences)))
    return percent_novel, novel_conf

def sequence_similarity(seq_list):
    import Bio
    from Bio import pairwise2
    from Bio.Align import substitution_matrices
    similarity_score=[]
    matrix = substitution_matrices.load("BLOSUM62")
    seq_set = list(set(seq_list))
    for seq in seq_set[:len(seq_set)//2]: #grab half the list
        for seq2 in seq_set[len(seq_set)//2:]: #grab other half
            if len(seq2)==0 or len(seq)==0:
                similarity_score.append(0)
            else:
                similarity_score.append( pairwise2.align.globaldx(seq,seq2, matrix, score_only=True)/(len(seq)+len(seq2)) )
    return similarity_score

#STATISTICS CONFIDENCE INTERVAL FUNCTIONS
#from https://stackoverflow.com/questions/13059011/is-there-any-python-function-library-for-calculate-binomial-confidence-intervals
def binconf(p, n, c=0.95):
    '''
    Calculate binomial confidence interval based on the number of positive and
    negative events observed. Wilson Score

    Parameters
    ----------
    p: int
      number of positive events observed
    n: int
      number of negative events observed
    c : optional, [0,1]
      confidence percentage. e.g. 0.95 means 95% confident the probability of
      success lies between the 2 returned values

    Returns
    -------
    theta_low  : float
      lower bound on confidence interval
    theta_high : float
      upper bound on confidence interval
    '''
    p, n = float(p), float(n)
    N    = p + n

    if N == 0.0: return (0.0, 1.0)

    p = p / N
    z = normcdfi(1 - 0.5 * (1-c))

    a1 = 1.0 / (1.0 + z * z / N)
    a2 = p + z * z / (2 * N)
    a3 = z * math.sqrt(p * (1-p) / N + z * z / (4 * N * N))

    return (a1 * (a2 - a3), a1 * (a2 + a3))

def erfi(x):
    """Approximation to inverse error function"""
    a  = 0.147  # MAGIC!!!
    a1 = math.log(1 - x * x)
    a2 = (2.0 / (math.pi * a)+ a1 / 2.0)

    return (sign(x) * math.sqrt( math.sqrt(a2 * a2 - a1 / a) - a2 ))

def sign(x):
    if x  < 0: return -1
    if x == 0: return  0
    if x  > 0: return  1

def normcdfi(p, mu=0.0, sigma2=1.0):
    """Inverse CDF of normal distribution"""
    if mu == 0.0 and sigma2 == 1.0:
        return math.sqrt(2) * erfi(2 * p - 1)
    else:
        return mu + math.sqrt(sigma2) * normcdfi(p)


####### GRADIENT TROUBLESHOOTING #########
#this needs to be used right after a call to "backwards" has been initiated so that the gradient is there

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            print(n,p.grad)
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    layers = np.array(layers)
    ave_grads = np.array(ave_grads)
    fig = plt.figure(figsize=(12,6))
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.ylim(ymin=0, ymax=5e-3)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    return plt
