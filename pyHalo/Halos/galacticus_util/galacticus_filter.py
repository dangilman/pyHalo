from pyHalo.Halos.galacticus_util.galacticus_util import GalacticusUtil
import numpy as np

def nodedata_apply_filter(nodedata,nodefilter,galacticus_util = None):
    """Takes a dictionary with numpy arrays as values and apply as boolean filter to all.
     Returns a dictionary with same keys, but a filter applied to all arrays."""
    return {key:val[nodefilter] for key,val in nodedata.items()}

def nodedata_select_subhalo(nodedata,id, galacticus_util = None):
    """Selects a subhalo specified by an id, returns a dictionary of the subhalo's properties"""
    gutil = GalacticusUtil() if galacticus_util is None else galacticus_util
    idfilter = nodedata[nodedata[gutil.PARAM_NODE_ID] == id]
    return {key:val[idfilter][0] for key,val in nodedata.items()}

def nodedata_filter_tree(nodedata, treeindex,galacticus_util = None):
    """Returns a filter that excludes all nodes but nodes in the specified tree"""
    gutil = GalacticusUtil() if galacticus_util is None else galacticus_util
    return nodedata[gutil.PARAM_TREE_INDEX] == treeindex

def nodedata_filter_subhalos(nodedata,galacticus_util= None):
    """Returns a filter that excludes all but subhalos (excludes host halos)"""
    gutil = GalacticusUtil() if galacticus_util is None else galacticus_util
    return nodedata[gutil.PARAM_ISOLATED] == 0

def nodedata_filter_halos(nodedata,galacticus_util = None):
    """Returns a filter that excludes all but halo (excludes sub-halos)"""
    return np.logical_not(nodedata_filter_subhalos(nodedata))

def nodedata_filter_range(nodedata,range,key,galacticus_util=None):
    """Returns a filter that excludes nodes not within the given range for a specified parameter"""
    return (nodedata[key] > range[0]) & (nodedata[key] < range[1]) 

def nodedata_filter_virialized(nodedata,galacticus_util= None):
    """
    Returns a filter that excludes everything outside of the host halos virial radius
    WARNING: Current implementation only works if there is only one host halo per tree,
    IE we are looking at the last output from galacticus. 
    """
    gutil = GalacticusUtil() if galacticus_util is None else galacticus_util

    #Get radial position of halos
    rvec = np.asarray((nodedata[gutil.PARAM_X],nodedata[gutil.PARAM_Y],nodedata[gutil.PARAM_Z]))
    r = np.linalg.norm(rvec,axis=0)

    #Filter halos and get there virial radii
    filtered_halos = nodedata_filter_halos(nodedata)
    rv_halos = nodedata[gutil.PARAM_RADIUS_VIRIAL][filtered_halos]
    halo_output_n = nodedata[gutil.PARAM_TREE_ORDER]

    filter_virialized = r < rv_halos[halo_output_n]

    return filter_virialized

def nodedata_filter_r2d(nodedata,r2d_max,plane_normal,
                        galacticus_util= None):
    
    """
    Filters based on projected radii.
    """
    gutil = GalacticusUtil() if galacticus_util is None else galacticus_util
    
    r = np.asarray((nodedata[gutil.PARAM_X],nodedata[gutil.PARAM_Y],nodedata[gutil.PARAM_Z]))

    r2d = project_r2d(*r,plane_normal)

    return r2d < r2d_max

def project_r2d(x,y,z,plane_normal):
    """
    Takes in arrays of coordinates, calculates the projected radius on the plane. 
     

    :param x: An array of x coordinates
    :param y: An array of y coordinates
    :param z: An array of z coordinates
    :plane_normal: Normal vector of the plane to project onto
    """
    coords = np.asarray((x,y,z))

    #Reshape so if scalers are passed for x,y,z we do not encounter issue
    if coords.ndim == 1:
        coords.reshape((3,1))

    #convert to unit normal
    un = plane_normal / np.linalg.norm(plane_normal)

    #Project distance to plane
    #Projected distance to plane is sqrt(r.r - (r.un)^2)
    rdotr = np.linalg.norm(coords,axis=0)**2
    rdotun = np.dot(un,coords)
    return np.sqrt(rdotr - rdotun**2)
    




    



    

