import numpy as np

def two_halo_boost(z, delta_z, host_m200, zlens, lensing_mass_function_class):

    boost = 1.
    if lensing_mass_function_class._two_halo_term and z==zlens:

        rmax = lensing_mass_function_class._cosmo.T_xy(zlens - delta_z, zlens)
        boost = lensing_mass_function_class.two_halo_boost(host_m200, z, rmax=rmax)

    return boost

def powerlaw_normalization(z, delta_z, zlens, lensing_mass_function_class, rendering_args, volume_element_comoving):

    boost = two_halo_boost(z, delta_z, rendering_args['parent_m200'], zlens, lensing_mass_function_class)

    norm_dV = rendering_args['LOS_normalization'] * boost * lensing_mass_function_class.norm_at_z_density(z)

    return norm_dV * volume_element_comoving

def delta_function_normalization(z, delta_z, mass, mass_fraction, zlens, lensing_mass_function_class, rendering_args,
                                 volume_element_comoving):

    boost = two_halo_boost(z, delta_z, rendering_args['parent_m200'], zlens, lensing_mass_function_class)

    n_dV = lensing_mass_function_class.dNdV_comoving_deltaFunc(
        mass, mass_fraction
    )

    n = n_dV * volume_element_comoving * boost * rendering_args['LOS_normalization']
    n = np.random.poisson(n)

    return n
