# -*- coding: utf-8 -*-

from typing import Optional, Sequence, Union, Mapping

import scanpy as sc
from anndata import AnnData


_VarNames = Union[str, Sequence[str]]


def _wraps_plot_heatmap(wrapper):
    import inspect
    
    params = inspect.signature(sc.pl.heatmap).parameters.copy()
    wrapper_sig = inspect.signature(wrapper)
    wrapper_params = wrapper_sig.parameters.copy()
    
    params.pop("adata")
    params.pop("var_names")
    params.pop("groupby")
    wrapper_params.pop("kwds")
    
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


@_wraps_plot_heatmap
def heatmap_aucell(
        adata: AnnData,
        var_names: Union[_VarNames, Mapping[str, _VarNames]],
        groupby: Union[str, Sequence[str]],
        key: Optional[str] = None,
        **kwds,
        ):
    
    if key is None:
        key = 'aucell'
    
    if key not in adata.obsm:
        raise ValueError(f'Did not find .obsm["{key}"].')
    
    adata_tmp = AnnData(adata.obsm[key])
    adata_tmp.obs[groupby] = adata.obs[groupby]
    
    return sc.pl.heatmap(adata_tmp, var_names, groupby, **kwds)



















