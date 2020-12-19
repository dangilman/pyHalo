import numpy as np

def host_scaling_function(mhalo, z, k1 = 0.88, k2 = 1.7, k3 = -2):

    # interpolated from galacticus

    logscaling = k1 * np.log10(mhalo / 10**13) + k2 * np.log10(z + 0.5)

    return 10**logscaling

def normalization_sigmasub(sigma_sub, host_m200, zlens, kpc_per_asec_zlens, cone_opening_angle, plaw_index, m_pivot):

    a0_per_kpc2 = sigma_sub * host_scaling_function(host_m200, zlens)

    R_kpc = kpc_per_asec_zlens * (0.5 * cone_opening_angle)

    area = np.pi * R_kpc ** 2

    m_pivot_factor = m_pivot ** -(plaw_index+1)

    return area * a0_per_kpc2 * m_pivot_factor
