# -*- coding: utf-8 -*-

from typing import Optional, Union, List
from matplotlib.axes import Axes

import scanpy as sc
from anndata import AnnData


def _wraps_plot_spatial(wrapper):
    import inspect
    
    params = inspect.signature(sc.pl.spatial).parameters.copy()
    wrapper_sig = inspect.signature(wrapper)
    wrapper_params = wrapper_sig.parameters.copy()
    
    params.pop("adata")
    params.pop("basis")
    wrapper_params.pop("kwargs")
    
    wrapper_params.update(params)
    annotations = {
        k: v.annotation
        for k, v in wrapper_params.items()
        if v.annotation != inspect.Parameter.empty
    }
    if wrapper_sig.return_annotation is not inspect.Signature.empty:
        annotations["return"] = wrapper_sig.return_annotation

    wrapper.__signature__ = inspect.Signature(
        list(wrapper_params.values()), return_annotation=wrapper_sig.return_annotation
    )
    wrapper.__annotations__ = annotations
    
    return wrapper


@_wraps_plot_spatial
def spatial_aucell(
        adata: AnnData,
        key: Optional[str] = None,
        *,
        basis: str = "spatial",
        **kwargs,
        ) -> Union[Axes, List[Axes], None]:
    
    if key is None:
        key = 'aucell'
    
    if key not in adata.obsm:
        raise ValueError(f'Did not find .obsm["{key}"].')
    
    if basis not in adata.obsm:
        raise ValueError(f'Did not find .obsm["{basis}"].')
    
    if basis not in adata.uns:
        raise ValueError(f'Did not find .uns["{basis}"].')
    
    adata_tmp = AnnData(adata.obsm[key])
    adata_tmp.obsm[basis] = adata.obsm[basis]
    adata_tmp.uns[basis] = adata.uns[basis]
    
    return sc.pl.spatial(adata_tmp, basis=basis, **kwargs)



















