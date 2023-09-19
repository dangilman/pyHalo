"""
These functions define some preset halo mass function models that can be specified instead of individually specifying
a bunch of keyword arguments. This is intended for a quick and easy user interface. The definitions shown here can also
act as a how-to guide if one wants to explore more complicated descriptions of the halo mass function, as the models
presented here show what each keyword argument accepted by pyHalo does.
"""

from pyHalo.pyhalo import pyHalo
from pyHalo.Cosmology.geometry import Geometry
from pyHalo.Rendering.MassFunctions.mass_function_base import CDMPowerLaw
from pyHalo.Halos.HaloModels.TNFWemulator import TNFWSubhaloEmulator
from pyHalo.Halos.HaloModels.TNFWFromParams import TNFWFromParams
from pyHalo.Rendering.MassFunctions.density_peaks import ShethTormen
from pyHalo.Rendering.SpatialDistributions.uniform import LensConeUniform
from pyHalo.Rendering.SpatialDistributions.nfw import ProjectedNFW
from pyHalo.truncation_models import truncation_models
from pyHalo.concentration_models import preset_concentration_models
from pyHalo.mass_function_models import preset_mass_function_models
from pyHalo.single_realization import Realization
from copy import copy
from scipy.spatial.transform import Rotation
import numpy as np
from pyHalo.realization_extensions import RealizationExtensions
from pyHalo.utilities import de_broglie_wavelength, MinHaloMassULDM
from pyHalo.Halos.galacticus_util.galacticus_util import GalacticusUtil
from pyHalo.Halos.galacticus_util.galacticus_filter import nodedata_filter_subhalos,nodedata_filter_tree,nodedata_filter_virialized,nodedata_filter_range,nodedata_apply_filter
from lenstronomy.LensModel.Profiles.tnfw import TNFW
import h5py
from pyHalo.PresetModels.cdm import CDM
from pyHalo.PresetModels.external import CDMFromEmulator
from pyHalo.PresetModels.sidm import SIDM_core_collapse
from pyHalo.PresetModels.uldm import ULDM
from pyHalo.PresetModels.wdm import WDM, WDM_mixed, WDMGeneral


__all__ = ['preset_model_from_name']

def preset_model_from_name(name):
    """
    Retruns a preset_model function from a string
    :param name: the name of the preset model, should be the name of a function in this file
    :return: the function
    """
    if name == 'CDM':
        return CDM
    elif name == 'WDM':
        return WDM
    elif name == 'SIDM_core_collapse':
        return SIDM_core_collapse
    elif name == 'ULDM':
        return ULDM
    elif name == 'CDMEmulator':
        return CDMFromEmulator
    elif name == 'WDM_mixed':
        return WDM_mixed
    elif name == 'WDMGeneral':
        return WDMGeneral
    else:
        raise Exception('preset model '+ str(name)+' not recognized!')



def DMFromGalacticus(z_lens,z_source,galacticus_file,tree_index, kwargs_cdm,mass_range,mass_range_is_bound = True,
                     proj_plane_normal = None,include_field_halos=True,nodedata_filter = None,
                     galacticus_utilities = None,galactics_params_additional = None, proj_rotation_angles = None):
    """
    This generates a realization of halos using subhalo parameters provided from a specified tree in the galacticus file.
    See https://github.com/galacticusorg/galacticus/ for information on the galacticus galaxy formation model. 
    
    :param z_lens: main deflector redshift
    :param z_source: source redshift
    :param galacticus_file: str or h5py.File or dict[str,np.array]. If string reads the file as a given path as a galalacticus output file, 
        If h5py.File reads from the given h5py file.
        If dict[str,np.ndarray] treats as nodedata from galacticus output, NOTE: all np.ndarrays in dictionary should have matching size along dimension 0.
    :param tree_index:  The number of the tree to create a realization from. A single galacticus file contains multiple realizations (trees).
        NOTE: Trees output from galacticus are assigned a number starting at 1.
    :param mass_range: Specifies mass range to include subhalos within.
    :param mass_range_is_bound: If true subhalos are filtered bound mass, if false subhalos are filtered by infall mass.
    :param projection_normal: Projects the coordinates of subhalos from parameters onto a plane defined with the given (3D) normal vector.
        Use this to generate multiple realizations from a single galacticus tree. If is None, coordinates are projected on the x,y plane. 
    :param include_field_halos: If true includes feild halos, if false no feild halos are included.
    :param nodedata_filter: Expects a callable function that has input and output: (dict[str,np.ndarray], GalacticusUtil) -> np.ndarray[bool]
        ,subhalos are filtered based on the output np array. Defaults to None
    :param galacticus_params: Extra parameter(s) to read when loading in galacticus hdf5 file. Does not need to be set unless a
        nodedata_Filter function is passed that needs parameter that is not loaded by default.
    :param proj_rotation_angle: Alternative to providing proj_plane_normal argument. Expects a length 2 array: (theta, phi)
        with theta and phi being the angles in spherical coordinates of the normal vector of defining the plane to project coordinates onto.
    """
    gutil = GalacticusUtil() if galacticus_utilities is None else galacticus_utilities

    MPC_TO_KPC = 1E3

    #Only read needed parameters to save memory. 
    PARAMS_TO_READ_DEF = [gutil.X,gutil.Y,gutil.Z,gutil.TNFW_RHO_S,
                      gutil.TNFW_RADIUS_TRUNCATION,gutil.RVIR,
                      gutil.SCALE_RADIUS,gutil.MASS_BOUND,
                      gutil.MASS_BASIC,gutil.IS_ISOLATED]

    if isinstance(galactics_params_additional,str):
        galactics_params_additional = [galactics_params_additional]
    elif galactics_params_additional is None:
        galactics_params_additional = []
    else:
        galactics_params_additional = list(galactics_params_additional)

    params_to_read = galactics_params_additional + PARAMS_TO_READ_DEF
    
    # we create a realization of only line-of-sight halos by setting sigma_sub = 0.0
    # only include these halos if requested
    kwargs_cdm['sigma_sub'] = 0.0

    los_norm = 1 if include_field_halos else 0
    los_norm = kwargs_cdm.get("LOS_normalization") if not kwargs_cdm.get("LOS_normalization") is None else los_norm


    cdm_halos_LOS = CDM(z_lens, z_source, **kwargs_cdm,LOS_normalization=los_norm)
    
    # get lens_cosmo class from class containing LOS objects; note that this will work even if there are no LOS halos
    lens_cosmo = cdm_halos_LOS.lens_cosmo

    if isinstance(galacticus_file,str):
        nodedata = gutil.read_nodedata_galacticus(galacticus_file,params_to_read=params_to_read)
    elif isinstance(galacticus_file,h5py.File):
        nodedata = gutil.hdf5_read_galacticus_nodedata(galacticus_file,params_to_read=params_to_read)
    else:
        nodedata = galacticus_file


    #Set up for rotation of coordinates
    #Secify the normal vector for the plane we are projecting onto, if user specified ensure the vector is normalized
    nh = np.asarray((0,0,1)) if proj_plane_normal is None else proj_plane_normal / np.linalg.norm(proj_plane_normal)
    nh_x,nh_y,nh_z = nh

    theta = np.arccos(nh_z)
    phi = np.sign(nh_y) * np.arccos(nh_x/np.sqrt(nh_x**2 + nh_y**2)) if nh_x != 0 or nh_y != 0 else 0

    if not proj_rotation_angles is None:
        theta,phi = proj_rotation_angles


    #This rotation rotation maps the coordinates such that in the new coordinates zh = nh and the x,y coordinates after rotation
    #are the x-y coordinates in the plane 
    rotation = Rotation.from_euler("zyz",(0,theta,phi))

    coords = np.asarray((nodedata[gutil.X],nodedata[gutil.Y],nodedata[gutil.Z])) * MPC_TO_KPC

    #We would like define the x and y unit vectors, so we can project our coordinates
    xh_r = rotation.apply(np.array((1,0,0)))
    yh_r = rotation.apply(np.array((0,1,0)))

    kpc_per_arcsec_at_z = lens_cosmo.cosmo.kpc_proper_per_asec(z_lens)
    
    #Get the maximum r2d for the subhalo to be within the rendering volume
    r2dmax_kpc = (kwargs_cdm["cone_opening_angle_arcsec"] / 2) * kpc_per_arcsec_at_z

    coords_2d = np.asarray((np.dot(xh_r,coords),np.dot(yh_r,coords)))
    r2d_mag = np.linalg.norm(coords_2d,axis=0)

    filter_r2d = r2d_mag < r2dmax_kpc

    #Choose wether to filter by  bound / infall mass
    mass_key = gutil.MASS_BOUND if mass_range_is_bound else gutil.MASS_BASIC

    # Filter subhalos
    # We should exclude nodes that are not valid subhalos, such as host halo nodes and nodes that are outside the virial radius.
    # (Galacticus is not calibrated for nodes outside of virial radius)
    filter_subhalos = nodedata_filter_subhalos(nodedata,gutil)
    filter_virialized = nodedata_filter_virialized(nodedata,gutil)
    filter_mass = nodedata_filter_range(nodedata,mass_range,mass_key,gutil)
    filter_tree = nodedata_filter_tree(nodedata,tree_index,gutil)
    filter_extra = np.ones(filter_tree.shape,dtype=bool) if nodedata_filter is None else nodedata_filter(nodedata,gutil)

    filter_combined = filter_subhalos & filter_virialized & filter_mass & filter_tree & filter_extra & filter_r2d

    #Apply filter to nodedata and rvec
    nodedata = nodedata_apply_filter(nodedata,filter_combined)
    coords_2d = coords_2d[:,filter_combined]
    r2d_mag = r2d_mag[filter_combined]
    coords = coords[:,filter_combined]
    r3d_mag = np.linalg.norm(coords,axis=0)

    # Get rhos_s factor of 4 comes from the this galacticus output is
    # The density normalization of the underlying NFW halo at r = rs
    # Multiply by 4 to get the normalization for the halo profile
    rho_s = 4 * nodedata[gutil.TNFW_RHO_S] / (MPC_TO_KPC)**3


    rs  = nodedata[gutil.SCALE_RADIUS] * MPC_TO_KPC
    rt = nodedata[gutil.TNFW_RADIUS_TRUNCATION] * MPC_TO_KPC
    rv = nodedata[gutil.RVIR] * MPC_TO_KPC

    halo_list = []
    #Loop throught properties of each subhalos
    for n,m_infall in enumerate(nodedata[gutil.MASS_BASIC]):
        x,y = coords_2d[0][n], coords_2d[1][n]

        tnfw_args = {
            TNFWFromParams.KEY_RT:rt[n],
            TNFWFromParams.KEY_RS:rs[n],
            TNFWFromParams.KEY_RHO_S:rho_s[n],
            TNFWFromParams.KEY_RV:rv[n]
        }



        halo_list.append(TNFWFromParams(m_infall,x,y,r3d_mag[n],z_lens,True,lens_cosmo,tnfw_args))

    subhalos_from_params = Realization.from_halos(halo_list,lens_cosmo,kwargs_halo_model={},
                                                    msheet_correction=False, rendering_classes=None)

    return cdm_halos_LOS.join(subhalos_from_params)

        



        


    



    





        

