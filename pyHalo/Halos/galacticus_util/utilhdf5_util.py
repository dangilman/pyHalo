#!/usr/bin/env python

"""filename.py: Provides util for utilhdf5 classes"""

from typing import Any

from pyHalo.Halos.galacticus_util.utilhdf5_common import *
from pyHalo.Halos.galacticus_util.utilhdf5_util import ifisnone

def get_names_subgroup(g:h5py.Group)->set[str]:
    """Gets the names of the subgroups"""
    return {it for it in g.keys() if isinstance(g[it],h5py.Group)}

def get_names_subdataset(g:h5py.Group)->set[str]:
    """Gets the names of the subdatasets"""
    return {it for it in g.keys() if isinstance(g[it],h5py.Dataset)}

def read_dataset(d:h5py.Dataset)->np.ndarray:
    """Writes dataset to numpy.ndaray"""
    arr = np.zeros(d.shape,dtype=d.dtype)
    d.read_direct(arr)
    return arr

def read_datasets_group(g:h5py.Group, allowed:(Iterable[str] | None) = None, prefix:(str | None) = None)->dict[str,np.ndarray]:
    """
    Reads all datasets in the group.

    Parameters:
    g (h5py.Group): h5py group to read datasets from
    allowed (Iterable[str] | None) = None: If None, does nothing. If provided only read datasets that have names that are tabulated here.
    prefix (str | None) = None: If None does nothing, if provided adds a prefix to the name of each dataset

    Returns:
    (dict[str,np.ndarray]): An dictionary of read datasets indexed by the dataset name
    """
    dset_names = get_names_subdataset(g)

    if not allowed is None: dset_names = dset_names.intersection(set(allowed))

    _prefix = ifisnone(prefix,"")

    return {_prefix + name:read_dataset(g[name]) for name in dset_names}


