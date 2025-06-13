import numpy as np
from scipy.spatial.transform import Rotation
from pyHalo.Halos.HaloModels.TNFWFromParams import TNFWFromParams
from pyHalo.pyhalo import pyHalo
from pyHalo.Cosmology.geometry import Geometry
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

    #z_lens = np.mean(nodedata[gutil.PARAM_Z_LAST_ISOLATED][np.logical_not(nodedata_filter_subhalos(nodedata,gutil))])
    z_lens = 0.5

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

def DMSubhalosFromEmulator(z_lens, z_source, emulator_data_function, cone_opening_angle_arcsec=8.0):
    """
    Generate a pyHalo realization from an emulator trained on Galacticus simulations
    :param z_lens: main deflector redshift
    :param z_source: source redshift
    :param cone_opening_angle_arcsec: twice the radius in arcsec of the rendering area
    :param emulator_data_function: a callable function that on each call returns the properties of halos from the emulator
    :return: a pyHalo realization object
    """
    data = emulator_data_function()
    # FIRST WE CREATE AN INSTANCE OF PYHALO, WHICH SETS THE COSMOLOGY
    pyhalo = pyHalo(z_lens, z_source)
    # WE ALSO SPECIFY THE GEOMETRY OF THE RENDERING VOLUME
    geometry_type = 'DOUBLE_CONE'
    geometry = Geometry(pyhalo.cosmology, z_lens, z_source,
                        cone_opening_angle_arcsec, geometry_type)

    h = pyhalo.lens_cosmo.h
    MPC_TO_KPC = 1e3

    # Here, data = emulator_data from normalizing_flows.py
    massInfall = data[0]
    concentration = data[1]
    massBound = data[2]
    redshiftLastIsolated = data[3]
    lens_redshifts = np.array([z_lens] * len(massInfall))
    rt = data[4] # This has an MPC_TO_KPC conversion in the tabulate_params() function
    x_kpc = data[5] * MPC_TO_KPC
    y_kpc = data[6] * MPC_TO_KPC

    r_vir = ((3 * h * massInfall)/(4 * 200 * np.pi * pyhalo.lens_cosmo._nfw_param.rhoc_z(lens_redshifts)))**(1/3)
    r_vir = r_vir/h

    rs = r_vir/concentration
    ones = np.ones(len(concentration))
    rhos = h**2 * 200.0 / 3 * pyhalo.lens_cosmo._nfw_param.rhoc_z(lens_redshifts) * concentration**3 / (np.log(ones + concentration) - concentration / (ones + concentration))

    def tabulate_params():
        key_id = []
        for i in range(len(rhos)):
            key_id.append('TNFW')

        return {
            TNFWFromParams.KEY_RHO_S:       rhos / (MPC_TO_KPC)**3,
            TNFWFromParams.KEY_RS :         rs * MPC_TO_KPC,
            TNFWFromParams.KEY_RT :         rt * MPC_TO_KPC,
            TNFWFromParams.KEY_RV :         r_vir  * MPC_TO_KPC,
            TNFWFromParams.KEY_Z_INFALL:    redshiftLastIsolated,
            TNFWFromParams.KEY_ID :         key_id
        }
    tnfw_args_all = tabulate_params()
    halo_list = []
    for n,m_infall in enumerate(massInfall):
        tnfw_args = {key:val[n] for key,val in tnfw_args_all.items()}
        halo_list.append(TNFWFromParams(m_infall,x_kpc[n], y_kpc[n],None, z_lens,True,pyhalo.lens_cosmo,tnfw_args))

    subhalos_from_emulator = Realization.from_halos(halo_list, pyhalo.lens_cosmo, kwargs_halo_model= {},
                                                    msheet_correction=False, rendering_classes=None, geometry = geometry)

    return subhalos_from_emulator

def JointLOSWithEmulator(z_lens,
                           z_source,
                           emulator_data_function,
                           cone_opening_angle_arcsec,
                           preset_model_name_LOS,
                           **kwargs_preset_model):
    """
    Combine subhalos from an emulator with line-of-sight halos from another preset model
    :param z_lens: deflector redshift
    :param z_source: source redshift
    :param emulator_data_function: function for the emulator method
    :param cone_opening_angle_arcsec: twice the area of the rendering area in the main lens plane
    :param preset_model_name_LOS: the name of a preset model that generates line-of-sight halos
    :param kwargs_preset_model: keyword arguments for the line-of-sight preset model
    :return: a realization with subhalos generated from the emulator and line-of-sight halos generated by pyHalo
    """
    emulator_subhalos = DMSubhalosFromEmulator(z_lens, z_source, emulator_data_function, cone_opening_angle_arcsec)
    from pyHalo.preset_models import preset_model_from_name
    preset_model_function = preset_model_from_name(preset_model_name_LOS)
    los = preset_model_function(z_lens, z_source, **kwargs_preset_model)
    return emulator_subhalos.join(los)

