#!/usr/bin/env python

"""filename.py: implements HDF5Group Class"""
from __future__ import annotations
from pyHalo.Halos.galacticus_util.utilhdf5_common import *
from pyHalo.Halos.galacticus_util.utilhdf5_util import ifisnone
import numpy as np

#Issue with typing
#https://stackoverflow.com/questions/40925470/python-how-to-refer-a-class-inside-itself

class ImportedHDF5Group():
    """This class stores data read from GALACTICUS output files"""

    #This class can contain numpy arrays, itself and methods that return numpy arrays r GOutputGroups
    __data:(dict[str, (np.ndarray | 'ImportedHDF5Group')] | None) = None
    __attrs:(dict[str, (np.ndarray | float | int)] | None) = None
    __doc_str = None
    __name = None

    @property
    def name(self):
        return self.__name
    
    @property
    def attrs(self):
        return self.__attrs

    @property
    def keys(self):
        return self.__data.keys()
    
    @property
    def documentation(self)->str:
        return ifisnone(self.__doc_str,GalacticusHDF5Parameters.DOCSTR_NODOCUMENTATION)
    
    
    
    def tabulate_datasets(self, allowed_keys:(Iterable[str] | None) = None)->dict[str,np.ndarray]:
        """
        Returns a new dictionary with same keys but only datasets.
        
        Parameters:
        allowed_keys ((Iterable[str] | None) = None): If None does nothing, if set only tabulates datasets 
        with keys stored in this iterable variable. 

        Returns:
        dict[str,np.ndarray]: A dictionary of with same keys but only datasets.
        """
        return {key:dat for (key,dat) in self.__data.items() 
                if (isinstance(dat,np.ndarray) and ((allowed_keys is None) or (key in allowed_keys)))}

    def __init__(self,data:(dict[str, (np.ndarray | 'ImportedHDF5Group')] | None),name:str,doc_str:(str | None) = None,attrs:(dict[str, (np.ndarray | float | int)] | None) = None):
        self.__name = name
        self.__data = data
        self.__doc_str = doc_str
        self.__attrs = ifisnone(attrs,{})

    def __getitem__(self, key:str) -> (np.ndarray | 'ImportedHDF5Group'):
        return self.__data[key]