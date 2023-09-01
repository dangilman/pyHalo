#!/usr/bin/env python

"""filename.py: Implements hdf5Reader class"""

from typing import Iterable
import h5py
import numpy as np
from pyHalo.Halos.galacticus_util.utilhdf5_common import *
from pyHalo.Halos.galacticus_util.utilhdf5_groupreader import GroupReader
from pyHalo.Halos.galacticus_util.utilhdf5_importedhdf5group import ImportedHDF5Group
from pyHalo.Halos.galacticus_util.utilhdf5_importedhdf5group import ImportedHDF5Group
from pyHalo.Halos.galacticus_util import utilhdf5_galacticus_hdf5parameters as gconst
from pyHalo.Halos.galacticus_util.utilhdf5_util import *

class HDFFReader(GroupReader):    

    def __init__(self,name:str) -> None:
        self.__name = name
        super().__init__()

    def get_name(self) -> str:
        return self.__name

    def read_datasets(self, g: h5py.Group, **kwargs) -> dict[str, np.ndarray]:
        return read_datasets_group(g)
    
    def get_subgroupreaders(self, g: h5py.Group, **kwargs) -> Iterable[GroupReader]:
        subgroups = get_names_subgroup(g)

        return [HDFFReader(name) for name in subgroups]
    
    def read_attributes(self, g: h5py.Group, **kwargs) -> dict[str, np.ndarray | float | int] | None:
        attrs = {} 
        for key,val in g.attrs.items():
            attrs[key] = val

        return attrs
    
    @staticmethod
    def read_file(fname:str,**kwargs)->ImportedHDF5Group:
        with h5py.File(fname,"r") as f:
            reader = HDFFReader(fname)
            return reader.read_group(f,**kwargs)
        