# -*- coding: utf-8 -*-

from typing import Union, Optional
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import scanpy as sc
from anndata import AnnData


def _wraps_plot_scatter(wrapper):
    import inspect
    
    params = inspect.signature(sc.pl.embedding).parameters.copy()
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


@_wraps_plot_scatter
def embedding_aucell(
    adata: AnnData,
    basis: str,
    key: Optional[str] = None,
    **kwargs,
    ) -> Union[Figure, Axes, None]:
    
    if key is None:
        key = 'aucell'
    
    if key not in adata.obsm:
        raise ValueError(f'Did not find .obsm["{key}"].')
    
    if basis in adata.obsm.keys():
        basis_key = basis
    elif f"X_{basis}" in adata.obsm.keys():
        basis_key = f"X_{basis}"
    else:
        raise KeyError(
            f"Could not find entry in `obsm` for '{basis}'.\n"
            f"Available keys are: {list(adata.obsm.keys())}."
        )
    
    adata_tmp = AnnData(adata.obsm[key])
    adata_tmp.obsm[basis_key] = adata.obsm[basis_key]
    
    return sc.pl.embedding(adata_tmp, basis, **kwargs)



















