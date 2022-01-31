import re
import math
import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
    hits = 0
    misses = 0
    position_accs = np.zeros((2, max_len))
    for in_seq, out_seq in zip(input_sequences, output_sequences):
        if in_seq == out_seq:
            seq_accs.append(1)
        else:
            seq_accs.append(0)

        misses += abs(len(in_seq) - len(out_seq))
        for j, (token_in, token_out) in enumerate(zip(in_seq, out_seq)):
            if token_in == token_out:
                hits += 1
                position_accs[0,j] += 1
            else:
                misses += 1
            position_accs[1,j] += 1

    seq_acc = np.mean(seq_accs)
    token_acc = hits / (hits + misses)
    position_acc = []
    for i in range(max_len):
        position_acc.append(position_accs[0,i] / position_accs[1,i])
    return seq_acc, token_acc, position_acc

def calc_property_accuracies(pred_props, true_props, MCC=False):
    import math
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
    print("property accuracy :",(TN + TP),"/",(TN+TP+FP+FN),"=",(TN + TP) / (TN+TP+FP+FN) )
    if MCC:
        N = TN + TP + FN + FP
        S = (TP + FN) / N
        P = (TP + FP) / N
        if S!=1 and P!=1:
            MCC = ( (TP/N)-(S*P) )/ math.sqrt(P*S*(1-S)*(1-P))
        else: MCC="Data error Division by zero"
        print("MCC: ", MCC)
    return acc, MCC
   

def calc_entropy(sample):
    "Calculates Shannon information entropy for a set of input memories"
    es = []
    for i in range(sample.shape[1]):
        probs, bin_edges = np.histogram(sample[:,i], bins=1000, range=(-5., 5.), density=True)
        es.append(entropy(probs))
    return np.array(es)

####### ADDITIONAL METRIC CALCULATIONS #########

def load_gen(path):
    "Loads set of generated SMILES strings from path"
    smiles = pd.read_csv(path).SMILES.to_list()
    return smiles

def valid(smiles):
    "Returns valid SMILES (RDKit sanitizable) from a set of\
    SMILES strings"
    valid_smiles = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            pass
        else:
            try:
                Chem.SanitizeMol(mol)
                valid_smiles.append(smi)
            except ValueError:
                pass
    return valid_smiles

def calc_token_lengths(smiles):
    "Calculates the token lengths of a set of SMILES strings"
    lens = []
    for smi in smiles:
        smi = tokenizer(smi)
        lens.append(len(smi))
    return lens

def calc_MW(smiles):
    "Calculates the molecular weights of a set of SMILES strings"
    MWs = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        MWs.append(Descriptors.MolWt(mol))
    return MWs

def novel(smiles, train_smiles):
    "Returns novel SMILES strings that do not appear\
    in training set"
    set_smiles = set(smiles)
    set_train = set(train_smiles)
    novel_smiles = list(set_smiles - set_train)
    return novel_smiles

def unique(smiles):
    "Returns unique SMILES strings from set"
    unique_smiles = set(smiles)
    return list(unique_smiles)

def fingerprints(smiles):
    "Calculates fingerprints of a list of SMILES strings"
    fps = np.zeros((len(smiles), 1024))
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        fp = np.asarray(Morgan(mol, 2, 1024), dtype='uint8')
        fps[i,:] = fp
    return fps

def tanimoto_similarity(bv1, bv2):
    "Calculates Tanimoto similarity between two fingerprint bit vectors"
    mand = sum(moses_fp & train_fp)
    mor = sum(moses_fp | train_fp)
    return mand / mor

def pass_through_filters(smiles, data_dir='data'):
    """Filters SMILES strings based on method implemented in
    http://nlp.seas.harvard.edu/2018/04/03/attention.html"""
    _mcf = pd.read_csv('{}/mcf.csv'.format(data_dir))
    _pains = pd.read_csv('{}/wehi_pains.csv'.format(data_dir), names=['smarts', 'names'])
    _filters = [Chem.MolFromSmarts(x) for x in
                _mcf.append(_pains, sort=True)['smarts'].values]
    filtered_smiles = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        h_mol = Chem.AddHs(mol)
        filtered = False
        if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
            filtered = True
        if any(h_mol.HasSubstructMatch(smarts) for smarts in _filters):
            filtered = True
        if not filtered:
            filtered_smiles.append(smi)
    return filtered_smiles

def cross_diversity(set1, set2, bs1=5000, bs2=5000, p=1, agg='max',
                    device='cpu'):
    """
    Function for calculating the maximum average tanimoto similarity score
    between the generated set and the training set (this code is adapted from
    https://github.com/molecularsets/moses)
    """
    agg_tanimoto = np.zeros(len(set2))
    total = np.zeros(len(set2))
    set2 = torch.tensor(set2).to(device).float()
    for j in range(0, set1.shape[0], bs1):
        x_stock = torch.tensor(set1[j:j+bs1]).to(device).float()
        for i in range(0, set2.shape[0], bs2):
            y_gen = set2[i:i+bs2]
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                   y_gen.sum(0, keepdim=True) -tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p!= 1:
                jac = jac**p
            if agg == 'max':
                agg_tanimoto[i:i+y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i+y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i+y_gen.shape[1]] += jac.sum(0)
                total[i:i+y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    return 1 - np.mean(agg_tanimoto)


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
