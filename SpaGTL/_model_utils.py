# -*- coding: utf-8 -*-

import torch
import numpy as np


def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One hot a tensor of categories."""
    index = index.reshape((-1, 1))
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)


def get_params_dict(params_dict, genes):
    
    genes_all = params_dict['genes']
    
    if np.all(np.isin(genes, genes_all)) == False:
        raise ValueError('Genes ' + genes[~np.isin(genes, genes_all)] + ' not found')
    
    idx_subset = np.isin(genes_all, genes)
    
    params_dict_subset = {}
    params_dict_subset['QK'] = params_dict['QK'][idx_subset,:][:,idx_subset]
    params_dict_subset['layer1'] = params_dict['layer1'][:,idx_subset]
    params_dict_subset['layer2m'] = params_dict['layer2m']
    params_dict_subset['layer2v'] = params_dict['layer2v']
    params_dict_subset['layer3'] = params_dict['layer3']
    params_dict_subset['layer4'] = params_dict['layer4'][idx_subset,:]
    
    return params_dict_subset



















