#!/usr/bin/env python

"""filename.py: Implents GNodeDataReader class"""

from .utilhdf5_common import *
from pyHalo.Halos.galacticus_util.utilhdf5_groupreader import GroupReader
from pyHalo.Halos.galacticus_util.utilhdf5_importedhdf5group import ImportedHDF5Group
from pyHalo.Halos.galacticus_util.utilhdf5_util import *
from pyHalo.Halos.galacticus_util.utilhdf5_galacticus_hdf5parameters import GalacticusHDF5Parameters

class GNodeDataReader(GroupReader):
    """
    Class for reading the "Outputs/OutputN" group and "Outputs/OutputN/nodeData" group from a GALACTICUS output hdf5 file.
    """

    DOCSTR = "Galacticus Node data, keys are names of galacticus Node data datasets. No subgroups."

    __dict_ndgroup = None
    __dict_outputngroup = None

    __name = None

    def get_name(self) -> str:
        return self.__name
    
    def get_documentation(self,**kwargs) -> str | None:
        return self.DOCSTR
    
    def get_subgroupreaders(self, g: h5py.Group, **kwargs) -> Iterable['GroupReader']:
        return super().get_subgroupreaders(g,**kwargs)
    
    def read_datasets(self, g: h5py.Group, **kwargs) -> dict[str, np.ndarray]:
        #Reads all node data
        self.__dict_ndgroup = self.read_dsets_ndgroup(g,**kwargs)

        self.__dict_outputngroup = self.read_dsets_outputngroup(g,**kwargs)

        return self.__dict_ndgroup | self.__dict_outputngroup

    def read_dsets_outputngroup(self, g: h5py.Group, **kwargs) -> dict[str, np.ndarray]:
        #Reads datasets in the Output N group
        return read_datasets_group(g,prefix=GalacticusHDF5Parameters.PREFIX_TREE_DSET)
    
    def read_dsets_ndgroup(self, g: h5py.Group, **kwargs) -> dict[str, np.ndarray]:
        #Node data is stored in OutputN/nodeData
        ndgroup:h5py.Group = g[GalacticusHDF5Parameters.GGROUP_NODEDATA]
        
        return read_datasets_group(ndgroup,allowed=kwargs.get(GalacticusHDF5Parameters.KWARG_NODEDATA_ALLOWED))

    
    def get_properties(self, dsets: dict[str, np.ndarray], subgroups: dict[str, ImportedHDF5Group], **kwargs) -> dict[str, (np.ndarray | ImportedHDF5Group)]:
        #If no datasets return
        if len(dsets) == 0: return super().get_properties(dsets,subgroups,**kwargs)

        properties:dict[str,(np.ndarray | ImportedHDF5Group)] = {}

        #Record keys for tree data and nodedata datasets
        properties[GalacticusHDF5Parameters.PROPERTY_KEYS_NODEDATA] = np.array(list(self.__dict_ndgroup.keys()))
        properties[GalacticusHDF5Parameters.PROPERTY_KEYS_TREEDATA] = np.array(list(self.__dict_outputngroup.keys()))

        #Record the index of each tree
        counts = dsets[GalacticusHDF5Parameters.PREFIX_TREE_DSET + GalacticusHDF5Parameters.GPARAM_MERGERTREE_COUNT]
        treenums = dsets[GalacticusHDF5Parameters.PREFIX_TREE_DSET + GalacticusHDF5Parameters.GPARAM_MERGERTREE_INDEX]

        properties[GalacticusHDF5Parameters.PROPERTY_NODE_TREE] = np.concatenate([np.full(count,index) for (count,index) in zip(counts,treenums)])
        properties[GalacticusHDF5Parameters.PROPERTY_NODE_TREE_OUTPUTORDER] = np.concatenate([np.full(count,i) for (i,count) in enumerate(counts)])

        return properties


        

    def __init__(self, name) -> None:
        """Reads the Output / OutputN group, set name to the name of output being read. Eg Output1 is being read, set name to Output1"""
        self.__name = name
        super().__init__()

