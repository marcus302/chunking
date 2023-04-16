from itertools import product
from typing import Dict, Hashable, Sequence

import numpy as np
import xarray as xr
from dask.array import Array
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from xarray.core.indexes import PandasMultiIndex
from xarray.core.utils import multiindex_from_product_levels
from xarray.core.variable import IndexVariable, Variable


def _blockwise_stack_dask_graph(arr, output_chunks):
    name = "blockwise-stack-" + tokenize(arr)
    ichunks = tuple(range(len(chunks_v)) for chunks_v in arr.chunks)
    ochunks = tuple(range(len(chunks_v)) for chunks_v in output_chunks)

    dsk = {
        (name, *ochunk): (np.reshape, (arr.name, *ichunk), oshape)
        for ichunk, ochunk, oshape in zip(
            product(*ichunks), product(*ochunks), product(*output_chunks)
        )
    }

    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[arr])
    res = Array(graph, name, output_chunks, arr.dtype, meta=arr._meta)

    return res


def _blockwise_stack_var(
    var: Variable, input_dims: Sequence[str], output_dim: str
) -> Variable:
    other_dims = [d for d in var.dims if d not in input_dims]
    dim_order = other_dims + list(input_dims)
    reordered = var.transpose(*dim_order)

    if reordered.chunks is None:
        raise ValueError("Unnecessary check that is here to please mypy.")

    output_chunks = tuple(reordered.chunks[: len(other_dims)]) + (
        tuple(map(np.product, product(*reordered.chunks[len(other_dims) :]))),
    )
    new_data = _blockwise_stack_dask_graph(reordered.data, output_chunks)

    new_dims = reordered.dims[: len(other_dims)] + (output_dim,)

    return Variable(new_dims, new_data, var._attrs, var._encoding, fastpath=True)


def blockwise_stack(
    ds: xr.Dataset,
    input_dims: Sequence[str],
    output_dim: str,
    ds_chunks: Dict[str, int],
):
    # TODO: Ensure ds is uniformly chunked for all data variables and coordinates according
    # to ds_chunks. Data otherwise will be misaligned.

    new_variables: dict[Hashable, Variable] = {}

    for name, var in ds.variables.items():
        if name not in input_dims:
            if any(d in var.dims for d in input_dims):
                add_dims = [d for d in input_dims if d not in var.dims]
                vdims = list(var.dims) + add_dims
                shape = [ds.dims[d] for d in vdims]
                # Set_dims broadcasts var if necessary.
                exp_var = var.set_dims(vdims, shape)
                if exp_var.chunks is None:
                    raise ValueError("Blockwise stacking only supports Dask arrays.")
                exp_var = exp_var.chunk(
                    {k: v for k, v in ds_chunks.items() if k in exp_var.dims}
                )
                stacked_var = _blockwise_stack_var(exp_var, input_dims, output_dim)
                new_variables[name] = stacked_var
            else:
                new_variables[name] = var.copy(deep=False)

    levels = [ds.get_index(dim) for dim in input_dims]
    idx = multiindex_from_product_levels(levels, names=input_dims)
    new_variables[output_dim] = IndexVariable(output_dim, idx)

    coord_names = set(ds._coord_names) - set(input_dims) | {output_dim}

    indexes = {k: v for k, v in ds.xindexes.items() if k not in input_dims}
    indexes[output_dim] = PandasMultiIndex(idx, output_dim)

    new_ds = ds._replace_with_new_dims(
        new_variables, coord_names=coord_names, indexes=indexes
    )
    # Makes the result of this function unstackable.
    new_ds = new_ds.reset_index(output_dim).drop_vars(input_dims)

    return new_ds
