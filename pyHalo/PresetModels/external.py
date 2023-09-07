from pyHalo.Halos.HaloModels.TNFWemulator import TNFWSubhaloEmulator
from pyHalo.PresetModels.cdm import CDM
from pyHalo.single_realization import Realization


def galacticus_subhalos(zlens, zsource, galacticus_hdf5, mdef='TNFW', rmax_arcsec=3.0):

    """
    This module loads a realization of halos from galacticus and transforms it into a pyHalo-style Realization
    :param zlens: main deflector redshift
    :param zsource: source redshift
    :param galacticus_hdf5: an HDF5 file the contains galacticus output
    :param rmax_arcsec: the maximum radius out to which one wants to render subhalos [arcsec]
    :return:
    """
    pass



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
