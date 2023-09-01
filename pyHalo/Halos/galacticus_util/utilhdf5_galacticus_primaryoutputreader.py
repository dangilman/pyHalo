#!/usr/bin/env python

"""filename.py: Implements GPrimaryOutputReader class"""

from pyHalo.Halos.galacticus_util.utilhdf5_common import *
from pyHalo.Halos.galacticus_util.utilhdf5_groupreader import GroupReader
from pyHalo.Halos.galacticus_util.utilhdf5_importedhdf5group import ImportedHDF5Group
from pyHalo.Halos.galacticus_util.utilhdf5_galacticus_nodedatareader import GNodeDataReader
from pyHalo.Halos.galacticus_util.utilhdf5_util import *
import string


class GalacticusPrimaryOutputReader(GroupReader):
    """
    Class for reading the primary output of the GALACTICUS output file IE "Outputs".
    """

    def get_documentation(self, **kwargs) -> str | None:
        return "TODO: Document"
    
    def get_name(self) -> str:
        return GalacticusHDF5Parameters.GGROUP_OUTPUT_PRIMARY
    
    def read_datasets(self, g: h5py.Group, **kwargs) -> dict[str, np.ndarray]:
        return super().read_datasets(g,**kwargs)
    
    def get_subgroupreaders(self, g: h5py.Group, **kwargs) -> Iterable['GroupReader']:
        subgroups = get_names_subgroup(g)

        return [GNodeDataReader(name) for name in subgroups]
    
    def get_properties(self, dsets: dict[str, np.ndarray], subgroups: dict[str, ImportedHDF5Group], **kwargs) -> dict[str, (np.ndarray | ImportedHDF5Group)]:
        #if no subgroups, return
        if len(subgroups) == 0: return super().get_properties(dsets,subgroups,**kwargs)
        
        properties:dict[str, (np.ndarray | ImportedHDF5Group)] = {}
        
        group_numbers = np.array([int(name.strip(string.ascii_letters)) for name in subgroups], dtype=int)
        
        sorted = np.sort(group_numbers)

        #Store the first and last outut as these are very commonly acessed
        properties[GalacticusHDF5Parameters.PROPERTY_OUTPUTS_FIRST] = subgroups[f"{GalacticusHDF5Parameters.GGROUP_OUTPUT_PREFIX}{sorted[0]}"]
        properties[GalacticusHDF5Parameters.PROPERTY_OUTPUTS_FINAL] = subgroups[f"{GalacticusHDF5Parameters.GGROUP_OUTPUT_PREFIX}{sorted[-1]}"]

        return properties

