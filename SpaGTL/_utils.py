# -*- coding: utf-8 -*-

from anndata import AnnData

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from natsort import natsorted


def refine_spatial_domains(
    adata: AnnData,
    domain_key: str,
    spatial_key: str = 'spatial',
    n_neighbors: int = 6,
) -> None:
    '''
    Refine spatial domains based on spatial neighbors.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    domain_key
        The key to find the identified spatial domains in `adata.obs`.
    spatial_key
        The key to find the spatial corrdinates in `adata.obsm`.
    n_neighbors
        Number of spatial neighbors to refine spatial domains.
    
    Returns
    -------
    Updates ``adata`` with the following fields.
    
    .obs[domain_key+'refined']
        The refined spatial domains.
    '''
    
    y_pred = adata.obs[domain_key]
    coord = np.array(adata.obsm[spatial_key])
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(coord)
    distances, indices = nbrs.kneighbors(coord)
    indices = indices[:,1:]
    
    y_refined = pd.Series(index=y_pred.index, dtype='object')
    
    for i in range(y_pred.shape[0]):
        
        y_pred_count = y_pred[indices[i,:]].value_counts()
        
        if (y_pred_count[y_pred[i]] < n_neighbors/2) and (y_pred_count.max() >= n_neighbors/2):
            y_refined[i] = y_pred_count.idxmax()
        else:
            y_refined[i] = y_pred[i]
            
    adata.obs[domain_key+'_refined'] = pd.Categorical(
        values=y_refined,
        categories=natsorted(map(str, np.unique(y_refined))),
    )
    
    return None



















