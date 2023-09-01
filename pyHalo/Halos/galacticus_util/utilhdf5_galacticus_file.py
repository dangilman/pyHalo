#!/usr/bin/env python

"""filename.py: Implements GFile class"""

from pyHalo.Halos.galacticus_util.utilhdf5_common import *
from pyHalo.Halos.galacticus_util.utilhdf5_groupreader import GroupReader
from pyHalo.Halos.galacticus_util.utilhdf5_galacticus_primaryoutputreader import GalacticusPrimaryOutputReader 
from pyHalo.Halos.galacticus_util.utilhdf5_importedhdf5group import ImportedHDF5Group

class GalacticusFile(ImportedHDF5Group):
    """
    Stores GOutputGroups read from a GALACTICUS output file. 
    Inherits from GOutputGroup.
    """

    #Private variables
    __fname = None

    #Properties
    @property
    def fname(self)->str:
        return self.__fname
    
    def __init__(self,data: dict[str, (np.ndarray | ImportedHDF5Group)],fname:str):
        self.__fname = fname
        super().__init__(data,fname, GalacticusHDF5Parameters.DOCSTR_GFILE)