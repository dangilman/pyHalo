#!/usr/bin/env python

"""filename.py: Implements GalacticusFileReader class"""

from pyHalo.Halos.galacticus_util.utilhdf5_common import * #utilhdf5_dependencies import *
from pyHalo.Halos.galacticus_util.utilhdf5_groupreader import GroupReader
from pyHalo.Halos.galacticus_util.utilhdf5_galacticus_file import GalacticusFile
from pyHalo.Halos.galacticus_util.utilhdf5_galacticus_primaryoutputreader import GalacticusPrimaryOutputReader
from pyHalo.Halos.galacticus_util.utilhdf5_importedhdf5group import ImportedHDF5Group
from pyHalo.Halos.galacticus_util.utilhdf5_hdf5reader import HDFFReader
from pyHalo.Halos.galacticus_util.utilhdf5_util import get_names_subgroup


class GalacticusFileReader(GroupReader):
    """
    This class is a general purpose class for reading GALACTICUS output files. 
    GGroupReaders are used by this class to read specific output groups.
    """

    DEFAULTREADERS = [GalacticusPrimaryOutputReader()]
    """Default GGroupReaders to use if none are provided"""

    __readers:Iterable[GroupReader] = None
    __fname:str = None

    @staticmethod
    def read_file(fname:str, readers:(Iterable[GroupReader] | None) = None, **kwargs)->GalacticusFile:
        """
        Reads a GALACTICUS file at a specified path using provided readers or default readers.
        
        Parameters:
        fname (str): Path to GALACTICUS output file
        readers (Itrerable[GGroupReader] | None): Readers to read output groups of the galacticus file.
        If None defaults to GalacticusFileReader.DEFAULTREADERS .
        kwargs: Key word arguments to use when reading file.

        Returns:
        GFile: A class giving the data read from the GALACTICUS output file
        """
        r = GalacticusFileReader(fname,readers)
        return r.read(**kwargs)

    def get_name(self) -> str:
        """Returns selected file path"""
        return self.__fname

    def read(self,**kwargs)->GalacticusFile:
        """
        Reads file with path / readers given in constructor.
        You may also use the static method read_file() to read a galacticus file without initializing an object.
        """
        with h5py.File(self.__fname,"r") as f:
            return self.read_group(f,**kwargs)

    def read_group(self, g: h5py.Group, **kwargs) -> GalacticusFile:
        d = self.create_dictionary(g,**kwargs)

        return self.create_outputgroup(d,**kwargs)
    
    def create_outputgroup(self, dictionary: dict[str, (np.ndarray | ImportedHDF5Group)], **kwargs) -> GalacticusFile:
        return GalacticusFile(dictionary,self.__fname)

    def read_datasets(self, g: h5py.Group, **kwargs) -> dict[str, np.ndarray]:
        return super().read_datasets(g,**kwargs)
    
    def get_subgroupreaders(self, g: h5py.Group, **kwargs) -> Iterable['GroupReader']:
        subgroups = get_names_subgroup(g)
        readernames = [reader.get_name() for reader in self.__readers]

        toread = [name for name in subgroups if not name in readernames]
        extrareaders = [HDFFReader(name) for name in toread]

        return self.__readers + extrareaders
    
    def get_properties(self, dsets: dict[str, np.ndarray], subgroups: dict[str, ImportedHDF5Group], **kwargs) -> dict[str, (np.ndarray | ImportedHDF5Group)]:
        return super().get_properties(dsets, subgroups, **kwargs)

    def __init__(self,fname:str, readers:(Iterable[GroupReader] | None) = None) -> None:
        self.__fname = fname
        self.__readers = ifisnone(readers,self.DEFAULTREADERS)