# -*- coding: utf-8 -*-

from typing import Optional, Tuple
from anndata import AnnData

import torch
import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix

from tqdm import trange

from ._module import SpaGTL
from ._model_utils import one_hot, get_params_dict


def _run_SpaGTL(
    X: np.ndarray,
    n_epochs: int,
    n_hidden: int,
    n_latent: int,
    params_dict_use: Optional[dict],
    n_batch: int,
    batch_index: Optional[np.ndarray],
    n_covar: int,
    covar: Optional[np.ndarray],
    device: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    
    if device is None or device == 'cuda':
        if torch.cuda.is_available():
          device = 'cuda'
        else:
          device = 'cpu'
    
    device = torch.device(device)
    
    data_X = torch.Tensor(X).to(device)
    if batch_index is not None:
        batch_index = one_hot(torch.Tensor(batch_index).to(device), n_batch)
    
    if covar is not None:
        covar = torch.Tensor(covar).reshape((-1, n_covar)).to(device)
        if batch_index is not None:
            covar = torch.concat((covar, batch_index), axis=0)
    else:
        covar = batch_index
    
    model = SpaGTL(
        n_input=data_X.shape[1],
        n_covar=n_covar+n_batch,
        n_hidden=n_hidden,
        n_latent=n_latent,
    )
    
    if params_dict_use is not None:
        model.load_pretrained_params(params_dict_use)
    
    model.to(device)
    model.train(mode=True)
    
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3, eps=0.01, weight_decay=1e-6)
    
    pbar = trange(n_epochs)
    
    for epoch in pbar:
        
        optimizer.zero_grad()
        
        inference_outputs = model.inference(data_X)
        generative_outputs = model.generative(inference_outputs['z'], covar)
        QK_outputs = model.forward_attention(data_X, covar)
        
        loss = model.loss(data_X, inference_outputs, generative_outputs, QK_outputs, epoch/n_epochs)
        
        pbar.set_postfix_str(f'loss: {loss.item():.3e}')
        
        loss.backward()
        optimizer.step()
    
    model.eval()
    
    with torch.no_grad():
        inference_outputs = model.inference(data_X)
        generative_outputs = model.generative(inference_outputs['z'], covar)
        qz = inference_outputs['qz'].loc.detach().cpu().numpy()
        x4 = generative_outputs['x4'].detach().cpu().numpy()
        QK = model.attention.getQK()
    
    return qz, x4, QK


def run_SpaGTL(
    adata: AnnData,
    n_epochs: int = 1000,
    n_hidden: int = 128,
    n_latent: int = 10,
    params_dict: Optional[dict] = None,
    batch_key: Optional[str] = None,
    covar_key: Optional[str] = None,
    device: Optional[str] = None,
    copy: bool = False,
) -> Optional[AnnData]:
    '''
    Spatially aligned Graph Transfer Learning for spatial transcriptomics.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    n_epochs
        Number of epochs for training neural network. Default to 1000.
    n_hidden
        Number of neurons in the hidden layer. Default to 128.
    n_latent
        Number of neurons in the latent layer. Default to 10.
    params_dict
        The pretrained parameters for initialing the neural network.
        If not specified, the parameters in the neural network is randomly initialized.
    batch_key
        The key to retriving batch information in `adata.obs[batch_key]`.
        If not specified, the batch correction is not considered.
    covar_key
        The key to retriving covariates in `adata.obsm[covar_key]`.
        If not specified, the covariates is not considered.
    device
        The desired device for `PyTorch` computation. By default uses cuda if cuda is avaliable
        cpu otherwise.
    copy
        Return a copy instead of writing to ``adata``.
    
    Returns
    -------
    Depending on ``copy``, returns or updates ``adata`` with the following fields.
    
    .obsm['qz'] : :class:`~numpy.ndarray`
        The latent representation of gene expression.
    .varp['QK'] : :class:`~scipy.sparse.csr_matrix`
        The gene-by-gene relation matrix.
    .layers['x4'] : :class:`~numpy.ndarray`
        The denoised gene expression matrix.
    '''
    
    adata = adata.copy() if copy else adata
    
    if params_dict is not None:
        params_dict_use = get_params_dict(params_dict, adata.var_names.to_numpy())
    else:
        params_dict_use = None
    
    if batch_key is not None:
        batch_info = pd.Categorical(adata.obs[batch_key])
        n_batch = batch_info.categories.shape[0]
        batch_index = batch_info.codes.copy()
    else:
        n_batch = 0
        batch_index = None
    
    if covar_key is not None:
        if covar_key in adata.obs.keys():
            covar = adata.obs[covar_key].to_numpy()
            n_covar = 1
        elif covar_key in adata.obsm.keys():
            covar = np.array(adata.obsm[covar_key])
            n_covar = covar.shape[1]
    else:
        n_covar = 0
        covar = None
    
    
    qz, x4, QK = _run_SpaGTL(
        X=adata.X.toarray() if issparse(adata.X) else adata.X,
        n_epochs=n_epochs,
        n_hidden=n_hidden,
        n_latent=n_latent,
        params_dict_use=params_dict_use,
        n_batch=n_batch,
        batch_index=batch_index,
        n_covar=n_covar,
        covar=covar,
        device=device,
    )
    
    
    key_added = 'SpaGTL'
    qz_key = 'qz'
    x4_key = 'x4'
    
    adata.uns[key_added] = {}
    
    neighbors_dict = adata.uns[key_added]
    
    neighbors_dict['params'] = {}
    neighbors_dict['params']['method'] = 'umap'
    
    adata.obsm[qz_key] = qz
    adata.layers[x4_key] = csr_matrix(x4)
    
    adata.uns['QK'] = {}
    neighbors_var_dict = adata.uns['QK']
    neighbors_var_dict['connectivities_key'] = 'QK'
    neighbors_var_dict['distances_key'] = 'QK'
    adata.varp['QK'] = csr_matrix(QK)
    
    return adata if copy else None



















