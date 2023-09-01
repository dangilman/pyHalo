from pyHalo.Halos.galacticus_util.utilhdf5_galacticus_hdf5parameters import GalacticusHDF5Parameters as pnhdf
from pyHalo.Halos.galacticus_util.galacticus_parameters import GalacticusParameters as pn
import numpy as np

def nodedata_apply_filter(nodedata:dict[str,np.ndarray],filter:np.ndarray):
    """Takes a dictionary with numpy arrays as values and apply as boolean filter to all.
     Returns a dictionary with same keys, but a filter applied to all arrays."""

    return {key:val[filter] for key,val in nodedata.items()}

def nodedata_filter_tree(nodedata:dict[str,np.ndarray], treenum:int):
    """Returns a filter that excludes all nodes but nodes in the specified tree"""
    return nodedata[pnhdf.PROPERTY_NODE_TREE] == treenum

def nodedata_filter_subhalos(nodedata:dict[str,np.ndarray]):
    """Returns a filter that excludes all but subhalos (excludes host halos)"""
    return nodedata[pn.IS_ISOLATED] == 0

def nodedata_filter_halos(nodedata:dict[str,np.ndarray]):
    """Returns a filter that excludes all but halo (excludes sub-halos)"""
    return np.logical_not(nodedata_filter_subhalos(nodedata))

def nodedata_filter_massrange(nodedata:dict[str,np.ndarray],mass_range,mass_key=pn.MASS_BASIC):
    """Returns a filter that excludes nodes not within the given mass range"""
    return (nodedata[mass_key] > mass_range[0]) & (nodedata[mass_key] < mass_range[1]) 

def nodedata_filter_virialized(nodedata:dict[str,np.ndarray]):
    """
    Returns a filter that excludes everything outside of the host halos virial radius
    WARNING: Current implementation only works if there is only one host halo per tree,
    IE we are looking at the last output from galacticus. 
    """
    #Get radial position of halos
    rvec = np.asarray((nodedata[pn.X],nodedata[pn.Y],nodedata[pn.Z]))
    r = np.linalg.norm(rvec,axis=0)

    #Filter halos and get there virial radii
    filtered_halos = nodedata_filter_halos(nodedata)
    rv_halos = nodedata[pn.RVIR][filtered_halos]
    halo_output_n = nodedata[pnhdf.PROPERTY_NODE_TREE_OUTPUTORDER][filtered_halos]

    filter_virialized = np.zeros(nodedata[pn.X].shape,dtype=bool)

    for n,rv in zip(halo_output_n,rv_halos):
        filter_virialized = filter_virialized | (r < rv) & (nodedata[pnhdf.PROPERTY_NODE_TREE_OUTPUTORDER] == n)

    return filter_virialized





    



    

