import matplotlib.pyplot as plt
import numpy as np
from pyHalo.single_realization import realization_at_z
from lenstronomy.Util.analysis_util import azimuthalAverage
from pyHalo.utilities import multiplane_convergence

plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['xtick.major.width'] = 3.5
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.minor.size'] = 6
plt.rcParams['ytick.minor.size'] = 6
plt.rcParams['ytick.major.width'] = 3.5
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['font.family']

def plot_subhalo_spatial_distribution(realization, max_range_arcsec=3.0, ax=None, color='k', kwargs_plot={},
                                     log_mlow=6.0, log_mhigh=10.0, nbins=25):
    """
    Plots the azimuthally-averaged radial distribution of subhalos
    :param realization: an instance of Realization that contains subhalos
    :param ax: figure axis
    :param color:
    :param kwargs_plot:
    :param log_mlow:
    :param log_mhigh:
    :param nbins:
    :return:
    """

    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)
    x_arcsec, y_arcsec = [], []
    for halo in realization.halos:
        if halo.is_subhalo:
            if halo.mass >= 10**log_mlow and halo.mass < 10**log_mhigh:
                x_arcsec.append(halo.x)
                y_arcsec.append(halo.y)
    root2 = np.sqrt(2)
    rangex, rangey = (-max_range_arcsec/root2, max_range_arcsec/root2), (-max_range_arcsec/root2, max_range_arcsec/root2)
    h, binx, biny = np.histogram2d(x_arcsec, y_arcsec, bins=nbins, range=(rangex, rangey))
    radial_prof, r_bin = azimuthalAverage(h)
    arcsec_per_pixel = max_range_arcsec / len(r_bin)
    r_bin_arcsec = r_bin * arcsec_per_pixel
    ax.plot(r_bin_arcsec, radial_prof, color=color, **kwargs_plot)
    ax.set_xlabel(r'$r_{\rm{2D}} \ \left[\rm{arcsec}\right]$', fontsize=16)
    ax.set_ylabel('number density '+r'$\left[\rm{arcsec}^{-2}\right]$', fontsize=16)

def plot_halo_mass_function(realization, z_eval=None, ax=None, color='k', kwargs_plot={},
                                     log_mlow=6.0, log_mhigh=10.0, nbins=25):

    """
    Makes a log-log plot of the halo mass function for a given realization.

    :param realization:
    :param z_eval:
    :param ax:
    :param color:
    :param kwargs_plot:
    :param log_mlow:
    :param log_mhigh:
    :param nbins:
    :param show_errorbars:
    :return:
    """
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)

    xlabel = 'halo mass '+r'$\left[M_{\odot}\right]$'
    ylabel = r'$n\left(m\right)$'
    if z_eval is None:
        masses = realization.masses
    elif isinstance(z_eval, float) or isinstance(z_eval, int):
        real = realization_at_z(realization, z_eval)[0]
        masses = real.masses
    elif isinstance(z_eval, list):
        masses = []
        for halo in realization.halos:
            if halo.is_subhalo:
                continue
            if halo.z >= z_eval[0] and halo.z < z_eval[1]:
                masses.append(halo.mass)
    else:
        raise Exception('z_eval must be one of: a specific redshift (i.e. a floating point number), or an'
                        'interval z_eval = [z_min, z_max]')
    h, m = np.histogram(masses, bins=np.logspace(log_mlow, log_mhigh, nbins))
    ax.plot(m[0:-1], h, color=color, **kwargs_plot)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')

def plot_concentration_mass_relation(realization, z_eval, ax=None, color='k', kwargs_plot={},
                                     log_mlow=6.0, log_mhigh=10.0, nbins=20, show_errorbars=False):

    """
    Makes a log-log plot of the concentration-mass relation for a given realization.

    :param realization: an instance of the Realization class
    :param z_eval: the redshift at which to evaluate the mc relation; can be one of 'z_lens', a flaot,
    or an interval [z_min, z_max]
    :param ax: figure axis
    :param color: the color for the lines in the figure
    :param kwargs_plot: keyword arguments passed to plot()
    :param log_mlow: minimum halo mass
    :param log_mhigh: maximum halo mass
    :param nbins: number of mass bins along the x axis
    :param show_errorbars: bool; show the logarithmic scatter
    :return:
    """
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)

    if z_eval == 'z_lens':
        xlabel = 'infall halo mass ' + r'$\left[M_{\odot}\right]$'
        ylabel = r'$c\left(m\right)$'
        halo_mass_list = []
        halo_concentration_list = []
        for halo in realization.halos:
            if halo.is_subhalo:
                halo_mass_list.append(halo.mass)
                halo_concentration_list.append(halo.c)
    elif isinstance(z_eval, float) or isinstance(z_eval, int):
        xlabel = 'halo mass ' + r'$\left[M_{\odot}\right]$'
        ylabel = r'$c\left(m\right)$'
        halo_mass_list = []
        halo_concentration_list = []
        for halo in realization.halos:
            if halo.is_subhalo and halo.z == z_eval:
                halo_mass_list.append(halo.mass)
                halo_concentration_list.append(halo.c)
    elif isinstance(z_eval, list):
        xlabel = 'halo mass ' + r'$\left[M_{\odot}\right]$'
        ylabel = r'$c\left(m\right)$'
        halo_mass_list = []
        halo_concentration_list = []
        for halo in realization.halos:
            if halo.z >= z_eval[0] and halo.z < z_eval[1]:
                halo_mass_list.append(halo.mass)
                halo_concentration_list.append(halo.c)
    else:
        raise Exception('z_eval must be one of: z_lens, a specific redshift (i.e. a floating point number), or an'
                        'interval z_eval = [z_min, z_max]')

    halo_masses = np.array(halo_mass_list)
    halo_concentrations = np.array(halo_concentration_list)

    x = np.logspace(log_mlow, log_mhigh, nbins + 1)
    c = []
    c_error = []
    for i in range(0, len(x) - 1):
        cond1 = halo_masses >= x[i]
        cond2 = halo_masses < x[i + 1]
        inds = np.where(np.logical_and(cond1, cond2))[0]
        c_median = np.median(halo_concentrations[inds])
        c_std = np.std(halo_concentrations[inds])
        c.append(c_median)
        c_error.append(c_std)

    if show_errorbars:
        ax.errorbar(x[0:-1], c, yerr=c_error, fmt='none', color=color)
    ax.plot(x[0:-1], c, color=color, **kwargs_plot)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xscale('log')
    ax.set_yscale('linear')

def plot_subhalo_bound_mass(realization, ax=None, color='k', kwargs_plot={},
                            log_mlow=6.0, log_mhigh=10.0):
    """
    Plots the bound mass of subhalos defined as the mass inside the infall virial radius of a truncated
    profile. Note: the halo mass definition must have a truncation radius specified for this method

    :param realization: an instance of Realization
    :param ax: figure axis
    :param color: color for the lines
    :param kwargs_plot: keyword arguments passed to plot
    :param log_mlow: minimum infall halo mass along x axis
    :param log_mhigh: maximum infall halo mass along x axis
    :return:
    """
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)
    masses = []
    mass_bound_fraction = []
    for halo in realization.halos:
        if halo.is_subhalo:
            masses.append(halo.mass)
            mass_bound_fraction.append(halo.bound_mass)

    ax.scatter(masses, mass_bound_fraction, color=color, **kwargs_plot)
    ax.plot(masses, masses, color='0.5', lw=2, linestyle=':')
    ax.set_xlim(10**log_mlow, 10**log_mhigh)
    ax.set_xlabel('infall halo mass '+r'$\left[M_{\odot}\right]$', fontsize=15)
    ax.set_ylabel('subhalo mass '+r'$M\left(r < r_{200}\right)$', fontsize=15)
    ax.set_xscale('log')
    ax.set_yscale('log')


def plot_subhalo_mass_functon(realization, log_m_low=6.0, log_m_high=10.0, bound_mass_function=False, nbins=20,
                              n_bootstrap=10, ax=None, color='k', kwargs_plot={}, rescale_amp=1.0):
    """
    Makes a log-log plot of the infall mass versus the final bound mass of tidally stripped subhalos
    :param realization: an instance of Realization
    :param log_mlow: minimum infall halo mass along x axis
    :param log_mhigh: maximum infall halo mass along x axis
    :param bound_mass_function:
    :param nbins: number of bins along x axis
    :param n_bootstrap: number of bootstrap resampling iteratins to compute vertical error bars
    :param ax: figure axis
    :param color: color for the lines
    :param kwargs_plot: keyword arguments passed to plot
    :param rescale_amp: rescales the plotted mass function
    :return:
    """
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)
    masses = []
    for halo in realization.halos:
        if halo.is_subhalo:
            if bound_mass_function:
                masses.append(halo.bound_mass)
            else:
                masses.append(halo.mass)
    h = np.empty((n_bootstrap, nbins))
    for i in range(0, n_bootstrap):
        inds = np.random.randint(0, len(masses), len(masses))
        _h, _x = np.histogram(np.log10(masses)[inds], bins=nbins, range=(log_m_low, log_m_high))
        h[i, :] = _h
    median = np.median(h, axis=0)
    standard_dev = np.std(h, axis=0)
    logm = _x[1:] - (_x[1] - _x[0]) / 2
    ax.errorbar(logm, rescale_amp * median, yerr=standard_dev, fmt='none', color=color)
    ax.plot(logm, rescale_amp * median, color=color, **kwargs_plot)
    ax.set_ylim(median[-1] / 2, median[0] * 2)
    ax.set_xlabel('infall halo mass ' + r'$\left[\log_{10} M_{\odot}\right]$', fontsize=15)
    ax.set_ylabel(r'$n\left(m\right)$', fontsize=15)
    ax.set_yscale('log')
    return (logm, median, standard_dev)

def plot_subhalo_concentration_versus_bound_mass(realization, ax=None, color='k', kwargs_plot={},
                                                 log_mlow=6.0, log_mhigh=10.0):
    """
    Plots the bound mass of subhalos, defined as the mass inside the infall virial radius of a truncated
    profile, versus the infall concentration of the halo.
    Note: the halo mass definition must have a truncation radius specified for this method

    :param realization: an instance of Realization
    :param ax: figure axis
    :param color: color for the lines
    :param kwargs_plot: keyword arguments passed to plot
    :param log_mlow: minimum infall halo mass along x axis
    :param log_mhigh: maximum infall halo mass along x axis
    :return:
    """
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)

    infall_concentration = []
    bound_mass_fraction = []
    for halo in realization.halos:
        if halo.is_subhalo:
            if halo.mass >= 10 ** log_mlow and halo.mass < 10 ** log_mhigh:
                infall_concentration.append(halo.c)
                bound_mass_fraction.append(halo.bound_mass / halo.mass)
    ax.scatter(infall_concentration, bound_mass_fraction, color=color, **kwargs_plot)
    ax.set_xlabel('infall concentration', fontsize=15)
    ax.set_ylabel(r'$\log_{10} \frac{M_{\rm{bound}}}{M_{\rm{infall}}}$', fontsize=22)
    # ax.set_xscale('log')
    ax.set_yscale('log')

def plot_subhalo_infall_time_versus_bound_mass(realization, ax=None, color='k', kwargs_plot={},
                                                 log_mlow=6.0, log_mhigh=10.0):
    """
    Plots the time since infall of subhalos versus the infall bound mass of the subhalo.
    Note: the halo mass definition must have a truncation radius specified for this method

    :param realization: an instance of Realization
    :param ax: figure axis
    :param color: color for the lines
    :param kwargs_plot: keyword arguments passed to plot
    :param log_mlow: minimum infall halo mass along x axis
    :param log_mhigh: maximum infall halo mass along x axis
    :return:
    """
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)

    infall_time = []
    bound_mass_fraction = []
    for halo in realization.halos:
        if halo.is_subhalo:
            if halo.mass >= 10 ** log_mlow and halo.mass < 10 ** log_mhigh:
                infall_time.append(halo.time_since_infall)
                bound_mass_fraction.append(halo.bound_mass / halo.mass)
    ax.scatter(infall_time, bound_mass_fraction, color=color, **kwargs_plot)
    ax.set_xlabel('time since infall [Gyr]', fontsize=15)
    ax.set_ylabel(r'$\log_{10} \frac{M_{\rm{bound}}}{M_{\rm{infall}}}$', fontsize=22)
    # ax.set_xscale('log')
    ax.set_yscale('log')

def plot_bound_mass_histogram(realization, ax=None, color='k', kwargs_plot={},
                              log_mlow=6.0, log_mhigh=10.0):
    """
    Plots the bound mass of subhalos defined as the mass inside the infall virial radius of a truncated
    profile.

    Note: the halo mass definition must have a truncation radius specified

    :param realization: an instance of Realization
    :param ax: figure axis
    :param color: color for the lines
    :param kwargs_plot: keyword arguments passed to plot
    :param log_mlow: minimum infall halo mass
    :param log_mhigh: maximum infall halo mass
    :return:
    """
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)

    bound_mass_fraction = []
    for halo in realization.halos:
        if halo.is_subhalo:
            if halo.mass >= 10 ** log_mlow and halo.mass < 10 ** log_mhigh:
                bound_mass_fraction.append(halo.bound_mass / halo.mass)
    ax.hist(np.log10(bound_mass_fraction), color=color, density=True, **kwargs_plot)
    ax.set_xlabel(r'$\log_{10} \frac{M_{\rm{bound}}}{M_{\rm{infall}}}$', fontsize=22)

def plot_truncation_radius_histogram(realization, subhalos_only=True, ax=None, color='k', kwargs_plot={},
                                   log_mlow=6.0, log_mhigh=10.0):
    """
    Plots a histogram of the truncation radius of a halo divided by the scale radius.

    Note: the halo mass definition must have a truncation radius specified for this method

    :param realization: an instance of Realization
    :param subhalos_only: bool; show only subhalos
    :param ax: figure axis
    :param color: color for the lines
    :param kwargs_plot: keyword arguments passed to plot
    :param log_mlow: minimum infall halo mass
    :param log_mhigh: maximum infall halo mass
    :return:
    """
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)

    tau = []
    for halo in realization.halos:
        if subhalos_only and halo.is_subhalo is False: continue
        if halo.mass >= 10 ** log_mlow and halo.mass < 10 ** log_mhigh:
            params = halo.params_physical
            rt_over_rs = params['r_trunc_kpc'] / params['rs']
            tau.append(rt_over_rs)
    ax.hist(np.log10(tau), color=color, density=True, **kwargs_plot)
    ax.set_xlabel(r'$\frac{r_t}{r_s}$', fontsize=15)


def plot_multiplane_convergence(realization, ax=None, npix=100, cone_opening_angle_arcsec=2.5, lens_model_list_macro=None,
                                kwargs_lens_macro=None, redshift_list_macro=None, cmap='bwr', vmin_max=0.05,
                                subtract_mean_kappa=False, show_critical_curve=True, grid_scale_crit_curve=0.025):
    """

    :param realization: an instance of Realization
    :param ax: a matplotlib axes
    :param npix: number of pixels per axis
    :param cone_opening_angle_arcsec: the size of each axis in arcsec
    :param lens_model_list_macro: a list of lens models that desceibe the main deflector
    :param kwargs_lens_macro: keyword arguments for the macro lens models
    :param redshift_list_macro: a list of redshifts for the macro lens models
    :param cmap: the name of a matploblib colormap
    :param vmin_max: color scale normalization for the convergence map
    :param subtract_mean_kappa: bool; subtract the average convergence in the area for visualization purposes
    :param show_critical_curve: bool; plots the critical curve
    :return: matplotlib axis instance
    """

    if ax is None:
        fig = plt.figure()
        fig.set_size_inches(6, 6)
        ax = plt.subplot(111)

    delta_kappa, lens_model, kwargs_lens, _, _ = multiplane_convergence(realization, cone_opening_angle_arcsec, npix, lens_model_list_macro,
                                         kwargs_lens_macro, redshift_list_macro)
    if subtract_mean_kappa:
        delta_kappa -= np.mean(delta_kappa)
    ax.imshow(delta_kappa, extent=[-cone_opening_angle_arcsec/2, cone_opening_angle_arcsec/2, -cone_opening_angle_arcsec/2, cone_opening_angle_arcsec/2],
              cmap=cmap, vmin=-vmin_max, vmax=vmin_max, origin='lower')

    if show_critical_curve:
        from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
        ext = LensModelExtensions(lens_model)
        ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list = ext.critical_curve_caustics(
            kwargs_lens, grid_scale=grid_scale_crit_curve, compute_window=cone_opening_angle_arcsec)
        ax.plot(ra_crit_list[0], dec_crit_list[0], color='k')

    return ax

