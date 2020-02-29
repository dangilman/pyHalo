import numpy as np

def host_scaling_function(mhalo, z, k1 = 0.88, k2 = 1.7, k3 = -2):

    # interpolated from galacticus

    logscaling = k1 * np.log10(mhalo * 10**-13) + k2 * np.log10(z + 0.5)

    return 10**logscaling

def norm_AO_from_sigmasub(sigma_sub, parent_m200, zlens, kpc_per_asec_zlens, cone_opening_angle, plaw_index, m_pivot=10**8):

    a0_per_kpc2 = sigma_sub * host_scaling_function(parent_m200, zlens)
    return norm_A0_from_a0area(a0_per_kpc2, kpc_per_asec_zlens,
                               cone_opening_angle, plaw_index, m_pivot)

def norm_A0_from_a0area(a0_per_kpc2, kpc_per_asec_zlens, cone_opening_angle, plaw_index, m_pivot=10**8):

    R_kpc = kpc_per_asec_zlens * (0.5 * cone_opening_angle)

    area = np.pi * R_kpc ** 2

    return a0_per_kpc2 * m_pivot ** (-plaw_index-1) * area

def convert_fsub_to_norm(f_sub, m_host, zhost, rein_arcsec, cone_opening_angle, kpc_per_asec_zlens, plaw_index, mlow,
                         mhigh, mpivot=10**8):


    power = 2+plaw_index
    R_kpc = kpc_per_asec_zlens * (0.5 * cone_opening_angle)
    #R_kpc = kpc_per_asec_zlens * rein_arcsec

    area = np.pi * R_kpc ** 2

    integral = (mpivot/power) * ((mhigh/mpivot)**power - (mlow/mpivot)**power)

    m_sub_scaled = f_sub * m_host * host_scaling_function(m_host, zhost)

    sigma_sub = m_sub_scaled / integral / area

    return sigma_sub

def norm_constant_per_squarearcsec(n_per_arcsecsquare, kpc_per_asec_zlens, cone_opening_angle, plaw_index):

    a0_per_kpc2 = n_per_arcsecsquare * kpc_per_asec_zlens ** -2
    return norm_A0_from_a0area(a0_per_kpc2, kpc_per_asec_zlens,
                               cone_opening_angle, plaw_index)
