#!/usr/bin/env python

"""filename.py: Implements GroupReader class"""

import h5py
import numpy as np
from pyHalo.Halos.galacticus_util.utilhdf5_importedhdf5group import ImportedHDF5Group
from typing import Iterable,Type


class GroupReader():
    """
    Abstract class for reading output groups from GALACTICUS hdf5 files.
    Inherit this class and override it's methods to read specific output groups.
    """



    def read_group(self,g:h5py.Group,**kwargs)->ImportedHDF5Group:
        """
        Reads group g. Calls create dictionary to get dictionary of read
        datasets / groups, then calls create_outputgroup().

        Parameters:
        g (h5py.Group): The group to be read
        kwargs: Parameters to use when reading from file

        Returns:
        (Any): Any output from reading group
        """
        dictionary = self.create_dictionary(g,**kwargs)

        attrs = self.read_attributes(g,**kwargs)

        return self.create_outputgroup(dictionary,attrs)

    
    def create_dictionary(self,g:h5py.Group,**kwargs)->dict[str,(np.ndarray | ImportedHDF5Group)]:
        """Calls read_datasets, get_subgroupreaders, and get_properties. Creates a dictionary from combined results. Can be overriden."""
        readdsets = self.read_datasets(g,**kwargs)

        readers = self.get_subgroupreaders(g,**kwargs)

        subgroups = {r.get_name(): r.read_group(g[r.get_name()],**kwargs) for r in readers}
    
        properties = self.get_properties(readdsets,subgroups,**kwargs)

        return readdsets | subgroups | properties
    
    def create_outputgroup(self,dictionary:dict[str, (np.ndarray | ImportedHDF5Group)],attrs:(dict[str,(np.ndarray | float | int)] | None), **kwargs)->ImportedHDF5Group:
        """Creates an output group based on the read datasets / groups and documentation. Can be overriden."""
        return ImportedHDF5Group(dictionary,name = self.get_name(),doc_str = self.get_documentation(**kwargs),attrs=attrs)

    def read_datasets(self,g:h5py.Group, **kwargs)->dict[str,np.ndarray]:
        """Override to read datasets. Default: Returns empty dictionary."""
        return {}
    
    def read_attributes(self, g:h5py.Group, **kwargs)->(dict[str,(np.ndarray | float | int)] | None):
        """Override to read attributes"""
        return {}

    def get_subgroupreaders(self,g:h5py.Group,**kwargs)->(Iterable['GroupReader']):
        """Overide to return GGroupReader objects that read subgroups. Default: Returns empty dictionary."""
        return {}

    def get_properties(self,dsets:dict[str,np.ndarray], subgroups:dict[str,ImportedHDF5Group],**kwargs)->dict[str,(np.ndarray | ImportedHDF5Group)]:
        """Overide to record properties of dsets / subgroups loaded. Default: Returns empty dictionary"""
        return {}
    
    def get_name(self)->str:
        """
        Abstract property. Must be overriden!
        The name of the current group.
        Default: Raises NotImplementedError
        
        Returns:
        The name of the group the group reader reads. 
        """
        raise NotImplementedError
    
    def get_documentation(self,**kwargs) -> str | None:
        return "No documentation implemented"
    

