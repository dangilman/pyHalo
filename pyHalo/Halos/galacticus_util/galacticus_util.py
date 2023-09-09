import h5py
import numpy as np
from typing import Iterable

class GalacticusUtil():
    """
    This class stores names of comonly used galacticus parameters
    """

    #Names of galacticus parameters
    #Position of the subhalo relative to the top level host halo
    X = "positionOrbitalX"
    """The GALACTICUS output parameter for the X coordinate of the subhalo relative to the main halo"""
    Y = "positionOrbitalY" 
    """The GALACTICUS output parameter for the Y coordinate of the subhalo relative to the main halo"""
    Z = "positionOrbitalZ"
    """The GALACTICUS output parameter for the Z coordinate of the subhalo relative to the main halo"""

    #Position of the subhalo relative to it's host halo or subhalo (not top level)
    RELX = "satellitePositionX"

    RELY = "satellitePositionY"

    RELZ = "satellitePositionZ"


    MASS_BOUND = "satelliteBoundMass"

    MASS_INFALL = MASS_BASIC = "basicMass"
    """The infall mass of the subhalo"""

    IS_ISOLATED = "nodeIsIsolated"

    HIERARCHYLEVEL = "nodeHierarchyLevel"

    RVIR = 'darkMatterOnlyRadiusVirial'

    SPHERE_RADIUS = "spheroidRadius"

    SPHERE_ANGLULARMOMENTUM = "spheroidAngularMomentum"

    SPHERE_MASS_STELLAR = "spheroidMassStellar"

    SPHERE_MASS_GAS = "spheroidMassGas"

    SCALE_RADIUS = "darkMatterProfileScaleRadius"

    DENSITY_PROFILE_RADIUS = "densityProfileRadius"

    DENSITY_PROFILE = "densityProfile"

    Z_LASTISOLATED = "redshiftLastIsolated"

    TNFW_RADIUS_TRUNCATION = "radiusTidalTruncationNFW"
    TNFW_RHO_S = "densityNormalizationTidalTruncationNFW"

    PARAM_TREE_ORDER = "custom_treeOutputOrder"
    PARAM_TREE_INDEX = "custom_treeIndex"
    


    DEF_TAB = [X,Y,Z,MASS_BOUND,MASS_BASIC,IS_ISOLATED,HIERARCHYLEVEL,RVIR]
    """Default galacticus parameters to include in tabulation"""

    HDF5_GROUP_OUTPUT_PRIMARY = "Outputs"
    """
    The name of the primary output group in a GALACTICUS output file. 
    This is the primary output group that contains nodedata at different output times
    """

    HDF5_GROUP_OUTPUT_N_PREFIX = "Output"
    """The name of the prefix of the "OutputN" groups in GALACTICUS output file"""

    HDF5_GROUP_NODEDATA = "nodeData"
    """The name of the hdf5 group containing nodedata."""

    HDF5_DSET_TREECOUNT = "mergerTreeCount"

    HDF5_DSET_TREEINDEX = "mergerTreeIndex"

    HDF5_DESET_TREESTART = "mergerTreeStartIndex"

    def hdf5_read_output_indicies(self,f):
        """
        Returns the numbers asigned to the various galacticus outputs.
        For if the galacicus file has the following groups: Outputs/Output1, Outputs/Output10
        this function will return [1,10]

        :param f: h5py.File read from Galacticus output
        """
        group_outputs = f[self.HDF5_GROUP_OUTPUT_PRIMARY]

        outputs = []

        trim = len(self.HDF5_GROUP_OUTPUT_N_PREFIX)
        for key in group_outputs.keys():
            try:
                if(key[:trim]) != self.HDF5_GROUP_OUTPUT_N_PREFIX:
                    continue
                outputs.append(int(key[trim:]))
            except(ValueError):
                pass
        
        return np.array(outputs)
    

    def hdf5_access_output_n_nodedata(self,f,output_n):
        """
        Returns the Outputs/OutputN/nodedata groups

        :param f: h5py.File read from Galacticus output
        :param output_n: The number coresponding to the output to be read
        """
        return f[self.HDF5_GROUP_OUTPUT_PRIMARY][f"{self.HDF5_GROUP_OUTPUT_N_PREFIX}{output_n}"][self.HDF5_GROUP_NODEDATA]


    def hdf5_read_dset(self,dset:h5py.Dataset):
        """
        Reads a hdf5 dataset into a numpy array
        """
        arr = np.zeros(dset.shape,dtype=dset.dtype)
        dset.read_direct(arr)
        return arr

    def hdf5_read_custom_nodedata(self,f,output_index:int):
        """
        To make analysis more convient, it is usefull to add custom datasets to those present in the Outputs/OutputN/nodedata.
        This function is used when read_nodedata_galacticus is called to create custom nodedata datasets for convienence.

        :param f: h5py.File read from Galacticus output
        :param output_index: The output to read indecies for.
        """
        group_outputn = f[self.HDF5_GROUP_OUTPUT_PRIMARY][f"{self.HDF5_GROUP_OUTPUT_N_PREFIX}{output_index}"]

        tree_index = self.hdf5_read_dset(group_outputn[self.HDF5_DSET_TREEINDEX])
        tree_start = self.hdf5_read_dset(group_outputn[self.HDF5_DESET_TREESTART])

        total_count = self.hdf5_read_nodecount_total(f,output_index)

        node_index = np.zeros(total_count)
        node_order = np.zeros(total_count)

        for n in range(1,len(tree_start)):
            start,stop = tree_start[n-1],tree_start[n]
            node_order[start:stop] = n - 1
            node_index[start:stop] = tree_index[n - 1] 

        node_order[stop:] = n
        node_index[stop:] = tree_index[n]

        return {GalacticusUtil.PARAM_TREE_INDEX:node_index,
                GalacticusUtil.PARAM_TREE_ORDER:node_order}
    
    def hdf5_read_nodecount_total(self,f,output_index):
        """
        Returns the total number of nodes at a given output.

        :param f: h5py.File read from Galacticus output
        :param output_index: The output index to read from.
        """

        group_outputn = f[self.HDF5_GROUP_OUTPUT_PRIMARY][f"{self.HDF5_GROUP_OUTPUT_N_PREFIX}{output_index}"]

        tree_count = self.hdf5_read_dset(group_outputn[self.HDF5_DSET_TREECOUNT])

        return np.sum(tree_count)
    
    def read_nodedata_galacticus(self,path, output_index = None,
                                params_to_read = None,
                                nodes_only=True):
        """
        Reads a galacticus output file at a given path, returns nodedata for a specified tree.
        Galacticus: https://github.com/galacticusorg/galacticus

        Note: if your build of galacticus has a different output formats you can inherit the GalacticusUtil class and modify
        parameter names / methods used when reading galacticus output. 

        :param path: Path to nodedata
        :param output_index: The index of the tree to read. If None defaults to the final output.
        :param params_to_read: A list of parameters to read, if None reads all parameters.
        :nodes_only: If true only reads nodedata entries that have entry for each node, if false reads all datasets in the nodedata group.
        """
        with h5py.File(path,"r") as f:
            return self.hdf5_read_galacticus_nodedata(f,output_index,params_to_read,nodes_only)


    def hdf5_read_galacticus_nodedata(self,f, output_index = None,
                                params_to_read = None,
                                nodes_only=True):
        """
        Reads a galacticus output given a h5py.File read from galacticus ouput
        Galacticus: https://github.com/galacticusorg/galacticus

        Note: if your build of galacticus has a different output formats you can inherit the GalacticusUtil class and modify
        parameter names / methods used when reading galacticus output. 

        :param f: A h5py.File read from galacticus output.
        :param output_index: The index of the tree to read. If None defaults to the final output.
        :param params_to_read: A list of parameters to read, if None reads all parameters.
        :nodes_only: If true only reads nodedata entries that have entry for each node, if false reads all datasets in the nodedata group.
        """

        #If no output is specified read the final output
        if output_index is None:

            outputs_ns = self.hdf5_read_output_indicies(f)

            #If no suitable outputs are found, return None

            if len(outputs_ns) == 0:

                return None
            
            output_index = np.max(outputs_ns)

        group_nodedata = self.hdf5_access_output_n_nodedata(f,output_index)

        nodedata = {}

        #Allow user to pass a single parameter to read as just a string

        if isinstance(params_to_read,str):

            params_to_read = [params_to_read] 

        nodecount = self.hdf5_read_nodecount_total(f,output_index)

        #Loop through datasets in nodedata

        for key in group_nodedata.keys():

            dset = group_nodedata[key]

            #Read dataset if it is a dataset and it is contained in the params_to_read variable or that variable is None
            if isinstance(dset,h5py.Dataset) and (params_to_read is None or key in params_to_read):
                if not nodes_only or (nodes_only and dset.shape[0] == nodecount):
                    nodedata[key] = self.hdf5_read_dset(dset)

        
        custom = self.hdf5_read_custom_nodedata(f,output_index)
        return nodedata | custom
