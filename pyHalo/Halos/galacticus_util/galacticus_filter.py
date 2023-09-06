from pyHalo.Halos.galacticus_util.galacticus_util import GalacticusUtil
import numpy as np

def nodedata_apply_filter(nodedata:dict[str,np.ndarray],filter:np.ndarray,galacticus_util:GalacticusUtil = None):
    """Takes a dictionary with numpy arrays as values and apply as boolean filter to all.
     Returns a dictionary with same keys, but a filter applied to all arrays."""
    gutil = GalacticusUtil() if galacticus_util is None else galacticus_util
    for key,val in nodedata.items():
        val[filter]
    
    return {key:val[filter] for key,val in nodedata.items()}

def nodedata_filter_tree(nodedata:dict[str,np.ndarray], treenum:int,galacticus_util:GalacticusUtil = None):
    """Returns a filter that excludes all nodes but nodes in the specified tree"""
    gutil = GalacticusUtil() if galacticus_util is None else galacticus_util
    return nodedata[gutil.PARAM_TREE_INDEX] == treenum

def nodedata_filter_subhalos(nodedata:dict[str,np.ndarray],galacticus_util:GalacticusUtil = None):
    """Returns a filter that excludes all but subhalos (excludes host halos)"""
    gutil = GalacticusUtil() if galacticus_util is None else galacticus_util
    return nodedata[gutil.IS_ISOLATED] == 0

def nodedata_filter_halos(nodedata:dict[str,np.ndarray],galacticus_util:GalacticusUtil = None):
    """Returns a filter that excludes all but halo (excludes sub-halos)"""
    gutil = GalacticusUtil() if galacticus_util is None else galacticus_util
    return np.logical_not(nodedata_filter_subhalos(nodedata))

def nodedata_filter_massrange(nodedata:dict[str,np.ndarray],mass_range,mass_key=GalacticusUtil.MASS_BASIC,
                              galacticus_util:GalacticusUtil = None):
    """Returns a filter that excludes nodes not within the given mass range"""
    gutil = GalacticusUtil() if galacticus_util is None else galacticus_util
    return (nodedata[mass_key] > mass_range[0]) & (nodedata[mass_key] < mass_range[1]) 

def nodedata_filter_virialized(nodedata:dict[str,np.ndarray],galacticus_util:GalacticusUtil = None):
    """
    Returns a filter that excludes everything outside of the host halos virial radius
    WARNING: Current implementation only works if there is only one host halo per tree,
    IE we are looking at the last output from galacticus. 
    """
    gutil = GalacticusUtil() if galacticus_util is None else galacticus_util

    #Get radial position of halos
    rvec = np.asarray((nodedata[gutil.X],nodedata[gutil.Y],nodedata[gutil.Z]))
    r = np.linalg.norm(rvec,axis=0)

    #Filter halos and get there virial radii
    filtered_halos = nodedata_filter_halos(nodedata)
    rv_halos = nodedata[gutil.RVIR][filtered_halos]
    halo_output_n = nodedata[gutil.PARAM_TREE_ORDER][filtered_halos]

    filter_virialized = np.zeros(nodedata[gutil.X].shape,dtype=bool)

    for n,rv in zip(halo_output_n,rv_halos):
        filter_virialized = filter_virialized | (r < rv) & (nodedata[gutil.PARAM_TREE_ORDER] == n)

    return filter_virialized


class ProjectionMode():
    """Different projection calculation modes"""
    KEY = "projection_mode"

    LOOP = 0
    """Use a loop to calculate the projection"""
    MATRIX = 1
    """Use matrix multiplication to calculate the projection"""
    EINSUM = 3

    DEFAULT = EINSUM
    """Default mode of projection"""

def nodedata_filter_r2d(nodedata:dict[str,np.ndarray],r2d_max:float,plane_normal:np.ndarray,projection_mode=ProjectionMode.DEFAULT,
                        galacticus_util:GalacticusUtil = None):
    gutil = GalacticusUtil() if galacticus_util is None else galacticus_util
    
    r = np.asarray((nodedata[gutil.X],nodedata[gutil.Y],nodedata[gutil.Z]))

    r2d = r2d_project(r,plane_normal,projection_mode)

    return r2d < r2d_max

def r2d_project(coords:np.ndarray,n:np.ndarray,projection_mode:int = ProjectionMode.DEFAULT)->np.ndarray:
    """
    Takes an numpy array of the form \n

    x1 x2 .. xn
    y1 y2 .. yn
    z1 z2 .. zn

    Ie (2,1) is the x2 corrdinate (3,2) is the y3 coordinate \n

    And the normal vector n, of the plane to project onto \n

    Projectets specified radius for all entries 

    takes kwarg - project_mode (int) -> Specify way to calculate projection, see ProjectionMode class
    
    NOTE COORDS CAN BE READ FROM GALACTICUSOUTPUT REDSHIFT USING util_tabulate(redshift)\n
    """
    #Exclude all nodes not in mass range and exclude all isolated nodes (main halos) 
    #For main halos is_isolated = 0.0f
    
    coords_select = coords.T

    r_2d_squared = np.zeros(coords_select.shape[0])

    #conver to unit normal
    un = n * (1/np.sqrt(np.dot(n,n)))

    #Project distance to plane
    #Projected distance to plane is sqrt(r.r - (r.un)^2)
    
    #Different (but equivalent) algorithms to calculating projections, using ProjectionMode.EINSUM is by far the fastest
    if projection_mode == ProjectionMode.EINSUM:
        #We have a matrix of rvectors [r1,r2,r3,...] 
        #We need to calculate R = [r1.r1,r2.r2,r3.r3,...]
        #Do this with einstein summation R_i = rij * rij 
        rdotr = np.einsum("ij,ij->i",coords_select,coords_select)

        rdotun = np.dot(coords_select,un)

        return np.sqrt(rdotr - rdotun**2)
    
    elif projection_mode == ProjectionMode.MATRIX:
        #Use matrix multiplication, slower because it has to calculate off diagonal entries
        rdotr =  np.diag(np.dot(coords_select,coords_select.transpose()))

        rdotun = np.dot(coords_select,un)

        return np.sqrt(rdotr - rdotun**2)

    elif projection_mode == ProjectionMode.LOOP:
        #Use simple loop
        #Calculate the square of the length of the vector projected onto plane for pair of coordinates
        for i,r in enumerate(coords_select):
            r_2d_squared[i] = np.dot(r,r) - (np.dot(r,un))**2

        #Take square root all at once - probably faster
        return np.sqrt(r_2d_squared)




    



    

