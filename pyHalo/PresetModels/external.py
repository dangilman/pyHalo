import numpy as np
from scipy.spatial.transform import Rotation
from pyHalo.Halos.HaloModels.TNFWemulator import TNFWSubhaloEmulator
from pyHalo.Halos.HaloModels.TNFWFromParams import TNFWFromParams
from pyHalo.PresetModels.cdm import CDM
from pyHalo.single_realization import Realization
from pyHalo.Halos.galacticus_util.galacticus_util import GalacticusUtil
from pyHalo.Halos.galacticus_util.galacticus_filter import nodedata_filter_subhalos,nodedata_filter_tree,nodedata_filter_virialized,nodedata_filter_range,nodedata_apply_filter
import h5py


def DMFromGalacticus(galacticus_hdf5,z_source,cone_opening_angle_arcsec,tree_index,log_mlow_galacticus,log_mhigh_galacticus,
                     mass_range_is_bound=True, proj_angle_theta=0, proj_angle_phi=0,
                     nodedata_filter=None, galacticus_utilities=None, galacticus_params_additional=None,
                     galacticus_tabulate_tnfw_params=None, preset_model_los="CDM", **kwargs_los):
    """
    This generates a realization of halos using subhalo parameters provided from a specified tree in the galacticus file.
    See https://github.com/galacticusorg/galacticus/ for information on the galacticus galaxy formation model.

    :param galacticus_hdf5: Galacticus output hdf5 file. If str loads file from specified path
    :param z_source: source redshift
    :param cone_opening_angle_arcsec: The opening angle of the cone in arc seconds
    :param tree_index: The number of the tree to create a realization from.
        A single galacticus file contains multiple realizations (trees).
        NOTE: Trees output from galacticus are assigned a number starting at 1.
    :param log_galacticus_mlow: The log_10 of the minimum mass subhalo to include in the realization.
        Subhalos are filtered by bound or infall mass as set by setting mass_range_is_bound.
    :param log_galacticus_mhigh: The log_10 of the minimum subhalo to include in the realization
        Subhalos are filtered by bound or infall mass as set by setting mass_range_is_bound.
    :param mass_range_is_bound: If true subhalos are filtered bound mass, if false subhalos are filtered by infall mass.
    :param proj_angle_theta: Specifies the theta angle (in spherical coordinates) of the normal vector for the plane to project subhalo coordinates on.
        Should be in the range [0,pi]
    :param proj_angle_theta: Specifies the theta angle (in spherical coordinates) of the normal vector for the plane to project subhalo coordinates on.
        Should be in the range [0, 2 * pi]
    :param nodedata_filter: Expects a callable function that has input and output: (dict[str,np.ndarray], GalacticusUtil) -> np.ndarray[bool]
        ,subhalos are filtered based on the output np array. Defaults to None. If provided overrides default filtering.
    :param galacticus_params: Extra parameters to read when loading in galacticus hdf5 file.
    :param galacticus_tabulate_params: Expects a callable function that has input and output: (dict[str,np.ndarray], GalacticusUtil) -> dict[str,np.ndarray]
        Should return a dictionary containing a numpy array of args for use in creation of tnfw halos.
    :param preset_model_los: Specifies the preset model to use when generating the LOS subhalos. Defaults to CDM

    :return: A realization from Galacticus halos

    NOTE:
    For galacticus output files: All trees should output at the same redshift. The final output should include only one host halo per tree.
    An example galacticus output and it's parameter file can be found in example_notebooks/data
    """
    #Avoid circular import
    from pyHalo.preset_models import preset_model_from_name

    gutil = GalacticusUtil() if galacticus_utilities is None else galacticus_utilities

    MPC_TO_KPC = 1E3

    #Only read needed parameters to save memory.
    PARAMS_TO_READ_DEF = [
                            gutil.PARAM_X,
                            gutil.PARAM_Y,
                            gutil.PARAM_Z,
                            gutil.PARAM_TNFW_RHO_S,
                            gutil.PARAM_TNFW_RADIUS_TRUNCATION,
                            gutil.PARAM_RADIUS_VIRIAL,
                            gutil.PARAM_RADIUS_SCALE,
                            gutil.PARAM_MASS_BOUND,
                            gutil.PARAM_MASS_INFALL,
                            gutil.PARAM_ISOLATED,
                            gutil.PARAM_Z_LAST_ISOLATED,
                            gutil.PARAM_CONCENTRATION
                         ]

    if galacticus_params_additional is None:
        galacticus_params_additional = []
    else:
        galacticus_params_additional = list(galacticus_params_additional)

    params_to_read = galacticus_params_additional + PARAMS_TO_READ_DEF

    if isinstance(galacticus_hdf5,str):
        nodedata = gutil.read_nodedata_galacticus(galacticus_hdf5,params_to_read=params_to_read)
    elif isinstance(galacticus_hdf5, h5py.Group):
        nodedata = gutil.hdf5_read_galacticus_nodedata(galacticus_hdf5,params_to_read=params_to_read)
    else:
        nodedata = galacticus_hdf5

    z_lens = np.mean(nodedata[gutil.PARAM_Z_LAST_ISOLATED][np.logical_not(nodedata_filter_subhalos(nodedata,gutil))])

    tree_index = int(np.round(tree_index))

    # we create a realization of only line-of-sight halos by setting sigma_sub = 0.0
    kwargs_los['sigma_sub'] = 0.0
    kwargs_los["cone_opening_angle_arcsec"] = cone_opening_angle_arcsec
    kwargs_los["z_lens"] = z_lens
    kwargs_los["z_source"] = z_source

    halos_LOS = preset_model_from_name(preset_model_los)(**kwargs_los)

    # get lens_cosmo class from class containing LOS objects; note that this will work even if there are no LOS halos
    lens_cosmo = halos_LOS.lens_cosmo

    theta,phi = proj_angle_theta,proj_angle_phi

    # This rotation rotation maps the coordinates such that in the new coordinates zh = nh
    # Then we apply this rotation to x and y unit vectors so we get the x and y unit vectors in the plane
    rotation = Rotation.from_euler("zyz",(0,theta,phi))

    coords = np.asarray((nodedata[gutil.PARAM_X],nodedata[gutil.PARAM_Y],nodedata[gutil.PARAM_Z])) * MPC_TO_KPC

    xh_r = rotation.apply(np.array((1,0,0)))
    yh_r = rotation.apply(np.array((0,1,0)))

    kpc_per_arcsec_at_z = lens_cosmo.cosmo.kpc_proper_per_asec(z_lens)

    r2dmax_kpc = (cone_opening_angle_arcsec / 2) * kpc_per_arcsec_at_z

    coords_2d = np.asarray((np.dot(xh_r,coords),np.dot(yh_r,coords)))
    r2d_mag = np.linalg.norm(coords_2d,axis=0)

    filter_r2d = r2d_mag < r2dmax_kpc
    mass_key = gutil.PARAM_MASS_BOUND if mass_range_is_bound else gutil.PARAM_MASS_INFALL
    mass_range = 10.0**np.asarray((log_mlow_galacticus,log_mhigh_galacticus))

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

    nodedata = nodedata_apply_filter(nodedata,filter_combined)
    coords_2d = coords_2d[:,filter_combined]
    r2d_mag = r2d_mag[filter_combined]
    coords = coords[:,filter_combined]
    r3d_mag = np.linalg.norm(coords,axis=0)

    def tabualate_params(nodedata,gutil):
        return {
            # The rhos_s factor of 4 comes from the this galacticus output is
            # The density normalization of the underlying NFW halo at r = rs
            # Multiply by 4 to get the normalization for the halo profile
            TNFWFromParams.KEY_RHO_S:       4 * nodedata[gutil.PARAM_TNFW_RHO_S] / (MPC_TO_KPC)**3,
            TNFWFromParams.KEY_RS :         nodedata[gutil.PARAM_RADIUS_SCALE] * MPC_TO_KPC,
            TNFWFromParams.KEY_RT :         nodedata[gutil.PARAM_TNFW_RADIUS_TRUNCATION] * MPC_TO_KPC,
            TNFWFromParams.KEY_RV :         nodedata[gutil.PARAM_CONCENTRATION] * nodedata[gutil.PARAM_RADIUS_SCALE] * MPC_TO_KPC,
            TNFWFromParams.KEY_Z_INFALL:    nodedata[gutil.PARAM_Z_LAST_ISOLATED],
            TNFWFromParams.KEY_ID :         nodedata[gutil.PARAM_NODE_ID]
        }

    galacticus_tabulate_tnfw_params = tabualate_params if galacticus_tabulate_tnfw_params is None else galacticus_tabulate_tnfw_params
    tnfw_args_all = galacticus_tabulate_tnfw_params(nodedata,gutil)

    halo_list = []
    for n,m_infall in enumerate(nodedata[gutil.PARAM_MASS_INFALL]):
        x,y = coords_2d[0][n], coords_2d[1][n]

        tnfw_args = {key:val[n] for key,val in tnfw_args_all.items()}

        halo_list.append(TNFWFromParams(m_infall,x,y,r3d_mag[n],z_lens,True,lens_cosmo,tnfw_args))

    subhalos_from_params = Realization.from_halos(halo_list,lens_cosmo,kwargs_halo_model={},
                                                    msheet_correction=False, rendering_classes=None)

    return halos_LOS.join(subhalos_from_params)

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
