import numpy as np
from scipy.spatial.transform import Rotation
from pyHalo.Halos.HaloModels.TNFWemulator import TNFWSubhaloEmulator
from pyHalo.Halos.HaloModels.TNFWFromParams import TNFWFromParams
from pyHalo.PresetModels.cdm import CDM
from pyHalo.single_realization import Realization
from pyHalo.Halos.galacticus_util.galacticus_util import GalacticusUtil
from pyHalo.Halos.galacticus_util.galacticus_filter import nodedata_filter_subhalos,nodedata_filter_tree,nodedata_filter_virialized,nodedata_filter_range,nodedata_apply_filter
from lenstronomy.LensModel.Profiles.tnfw import TNFW
import h5py


def DMFromGalacticus(z_lens,z_source,galacticus_hdf5,tree_index, kwargs_cdm,mass_range,mass_range_is_bound = True,
                     proj_plane_normal = None,include_field_halos=True,nodedata_filter = None,
                     galacticus_utilities = None,galacticus_params_additional = None, proj_rotation_angles = None):
    """
    This generates a realization of halos using subhalo parameters provided from a specified tree in the galacticus file.
    See https://github.com/galacticusorg/galacticus/ for information on the galacticus galaxy formation model. 
    
    :param z_lens: main deflector redshift
    :param z_source: source redshift
    :param galacticus_hdf5: Galacticus output hdf5 file. If str loads file from specified path
    :param tree_index:  The number of the tree to create a realization from. A single galacticus file contains multiple realizations (trees).
        NOTE: Trees output from galacticus are assigned a number starting at 1.
    :param mass_range: Specifies mass range to include subhalos within.
    :param mass_range_is_bound: If true subhalos are filtered bound mass, if false subhalos are filtered by infall mass.
    :param projection_normal: Projects the coordinates of subhalos from parameters onto a plane defined with the given (3D) normal vector.
        Use this to generate multiple realizations from a single galacticus tree. If is None, coordinates are projected on the x,y plane. 
    :param include_field_halos: If true includes field halos, if false no field halos are included.
    :param nodedata_filter: Expects a callable function that has input and output: (dict[str,np.ndarray], GalacticusUtil) -> np.ndarray[bool]
        ,subhalos are filtered based on the output np array. Defaults to None
    :param galacticus_params: Extra parameters to read when loading in galacticus hdf5 file.
    :param proj_rotation_angle: Alternative to providing proj_plane_normal argument. Expects a length 2 array: (theta, phi)
        with theta and phi being the angles in spherical coordinates of the normal vector of defining the plane to project coordinates onto.

    :return: A realization from Galacticus halos
    """
    gutil = GalacticusUtil() if galacticus_utilities is None else galacticus_utilities

    MPC_TO_KPC = 1E3

    #Only read needed parameters to save memory. 
    PARAMS_TO_READ_DEF = [gutil.PARAM_X,gutil.PARAM_Y,gutil.PARAM_Z,gutil.PARAM_TNFW_RHO_S,
                      gutil.PARAM_TNFW_RADIUS_TRUNCATION,gutil.PARAM_RADIUS_VIRIAL,
                      gutil.PARAM_RADIUS_SCALE,gutil.PARAM_MASS_BOUND,
                      gutil.PARAM_MASS_BASIC,gutil.PARAM_ISOLATED]

    if galacticus_params_additional is None:
        galacticus_params_additional = []
    else:
        galacticus_params_additional = list(galacticus_params_additional)

    params_to_read = galacticus_params_additional + PARAMS_TO_READ_DEF
    
    # we create a realization of only line-of-sight halos by setting sigma_sub = 0.0
    # only include these halos if requested
    kwargs_cdm['sigma_sub'] = 0.0

    los_norm = 1 if include_field_halos else 0
    los_norm = kwargs_cdm.get("LOS_normalization") if not kwargs_cdm.get("LOS_normalization") is None else los_norm


    cdm_halos_LOS = CDM(z_lens, z_source, **kwargs_cdm,LOS_normalization=los_norm)
    
    # get lens_cosmo class from class containing LOS objects; note that this will work even if there are no LOS halos
    lens_cosmo = cdm_halos_LOS.lens_cosmo

    if isinstance(galacticus_hdf5,str):
        nodedata = gutil.read_nodedata_galacticus(galacticus_hdf5,params_to_read=params_to_read)
    else:
        nodedata = gutil.hdf5_read_galacticus_nodedata(galacticus_hdf5,params_to_read=params_to_read)


    #Set up for rotation of coordinates
    #Specify the normal vector for the plane we are projecting onto, if user specified ensure the vector is normalized
    nh = np.asarray((0,0,1)) if proj_plane_normal is None else proj_plane_normal / np.linalg.norm(proj_plane_normal)
    nh_x,nh_y,nh_z = nh

    theta = np.arccos(nh_z)
    phi = np.sign(nh_y) * np.arccos(nh_x/np.sqrt(nh_x**2 + nh_y**2)) if nh_x != 0 or nh_y != 0 else 0

    if not proj_rotation_angles is None:
        theta,phi = proj_rotation_angles


    #This rotation rotation maps the coordinates such that in the new coordinates zh = nh and the x,y coordinates after rotation
    #are the x-y coordinates in the plane 
    rotation = Rotation.from_euler("zyz",(0,theta,phi))

    coords = np.asarray((nodedata[gutil.PARAM_X],nodedata[gutil.PARAM_Y],nodedata[gutil.PARAM_Z])) * MPC_TO_KPC

    #We would like define the x and y unit vectors, so we can project our coordinates
    xh_r = rotation.apply(np.array((1,0,0)))
    yh_r = rotation.apply(np.array((0,1,0)))

    kpc_per_arcsec_at_z = lens_cosmo.cosmo.kpc_proper_per_asec(z_lens)
    
    #Get the maximum r2d for the subhalo to be within the rendering volume
    r2dmax_kpc = (kwargs_cdm["cone_opening_angle_arcsec"] / 2) * kpc_per_arcsec_at_z

    coords_2d = np.asarray((np.dot(xh_r,coords),np.dot(yh_r,coords)))
    r2d_mag = np.linalg.norm(coords_2d,axis=0)

    filter_r2d = r2d_mag < r2dmax_kpc

    #Choose to filter by  bound / infall mass
    mass_key = gutil.PARAM_MASS_BOUND if mass_range_is_bound else gutil.PARAM_MASS_BASIC

    # Filter subhalos
    # We should exclude nodes that are not valid subhalos, such as host halo nodes and nodes that are outside the virial radius.
    # (Galacticus is not calibrated for nodes outside of virial radius)

    if nodedata_filter is None:
        filter_subhalos = nodedata_filter_subhalos(nodedata,gutil)
        filter_virialized = nodedata_filter_virialized(nodedata,gutil)
        filter_mass = nodedata_filter_range(nodedata,mass_range,mass_key,gutil)
        filter_tree = nodedata_filter_tree(nodedata,tree_index,gutil)

        filter_combined = filter_subhalos & filter_virialized & filter_mass & filter_tree & filter_r2d

    else:
        filter_combined = nodedata_filter(nodedata,gutil)

    #Apply filter to nodedata and rvec
    nodedata = nodedata_apply_filter(nodedata,filter_combined)
    coords_2d = coords_2d[:,filter_combined]
    r2d_mag = r2d_mag[filter_combined]
    coords = coords[:,filter_combined]
    r3d_mag = np.linalg.norm(coords,axis=0)

    # Get rhos_s factor of 4 comes from the this galacticus output is
    # The density normalization of the underlying NFW halo at r = rs
    # Multiply by 4 to get the normalization for the halo profile
    rho_s = 4 * nodedata[gutil.PARAM_TNFW_RHO_S] / (MPC_TO_KPC)**3


    rs  = nodedata[gutil.PARAM_RADIUS_SCALE] * MPC_TO_KPC
    rt = nodedata[gutil.PARAM_TNFW_RADIUS_TRUNCATION] * MPC_TO_KPC
    rv = nodedata[gutil.PARAM_RADIUS_VIRIAL] * MPC_TO_KPC

    halo_list = []
    #Loop thought properties of each subhalos
    for n,m_infall in enumerate(nodedata[gutil.PARAM_MASS_BASIC]):
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




def CDMFromEmulator(z_lens, z_source, emulator_input, kwargs_cdm):
    """
    This generates a realization of subhalos using an emulator of the semi-analytic modeling code Galacticus, and
     generates line-of-sight halos from a mass function parameterized as Sheth-Tormen.

    :param z_lens: main deflector redshift
    :param z_source: sourcee redshift
    :param emulator_input: either an array or a callable function

    if callable: a function that returns an array of
    1) subhalo masses at infall [M_sun]
    2) subhalo projected x position [kpc]
    3) subhalo projected y position [kpc]
    4) subhalo final_bound_mass [M_sun]
    5) subhalo concentrations at infall

    if not callable: an array with shape (N_subhalos 5) that contains masses, positions x, positions y, etc.

    Mass convention is m200 with respect to the critical density of the Universe at redshift Z_infall, where Z_infall is
    the infall redshift (not necessarily the redshift at the time of lensing).

    :param cone_opening_angle_arcsec: the opening angle of the double cone rendering volume in arcsec
    :param log_mlow: log10(minimum halo mass) rendered, or a function that returns log_mlow given a redshift
    :param log_mhigh: log10(maximum halo mass) rendered, or a function that returns log_mlow given a redshift
    :param LOS_normalization: rescaling of the line of sight halo mass function relative to Sheth-Tormen
    :param log_m_host: log10 host halo mass in M_sun
    :param kwargs_other: allows for additional keyword arguments to be specified when creating realization

    The following optional keywords specify a concentration-mass relation for field halos parameterized as a power law
    in peak height. If they are not set in the function call, pyHalo assumes a default concentration-mass relation from Diemer&Joyce
    :param c0: amplitude of the mass-concentration relation at 10^8
    :param log10c0: logarithmic amplitude of the mass-concentration relation at 10^8 (only if c0_mcrelation is None)
    :param beta: logarithmic slope of the mass-concentration-relation pivoting around 10^8
    :param zeta: modifies the redshift evolution of the mass-concentration-relation
    :param two_halo_contribution: whether to include the two-halo term for correlated structure near the main deflector
    :param kwargs_halo_mass_function: keyword arguments passed to the LensingMassFunction class
    (see Cosmology.lensing_mass_function)
    :return: a realization of CDM halos
    """

    # we create a realization of only line-of-sight halos by setting sigma_sub = 0.0
    kwargs_cdm['sigma_sub'] = 0.0
    cdm_halos_LOS = CDM(z_lens, z_source, **kwargs_cdm)
    # get lens_cosmo class from class containing LOS objects; note that this will work even if there are no LOS halos
    lens_cosmo = cdm_halos_LOS.lens_cosmo

    # now create subhalos from the specified properties using the TNFWSubhaloEmulator class
    halo_list = []
    if callable(emulator_input):
        subhalo_infall_masses, subhalo_x_kpc, subhalo_y_kpc, subhalo_final_bound_masses, \
        subhalo_infall_concentrations = emulator_input()
    else:
        subhalo_infall_masses = emulator_input[:, 0]
        subhalo_x_kpc = emulator_input[:, 1]
        subhalo_y_kpc = emulator_input[:, 2]
        subhalo_final_bound_masses = emulator_input[:, 3]
        subhalo_infall_concentrations = emulator_input[:, 4]

    for i in range(0, len(subhalo_infall_masses)):
        halo = TNFWSubhaloEmulator(subhalo_infall_masses[i],
                                   subhalo_x_kpc[i],
                                   subhalo_y_kpc[i],
                                   subhalo_final_bound_masses[i],
                                   subhalo_infall_concentrations[i],
                                   z_lens, lens_cosmo)
        halo_list.append(halo)

    # combine the subhalos with line-of-sight halos
    subhalos_from_emulator = Realization.from_halos(halo_list, lens_cosmo, kwargs_halo_model={},
                                                    msheet_correction=False, rendering_classes=None)
    return cdm_halos_LOS.join(subhalos_from_emulator)
