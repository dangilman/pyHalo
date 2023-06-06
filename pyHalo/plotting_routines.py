import matplotlib.pyplot as plt
import numpy as np

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
        halo_mass_list = []
        halo_concentration_list = []
        for halo in realization.halos:
            if halo.is_subhalo:
                halo_mass_list.append(halo.mass)
                halo_concentration_list.append(halo.c)
    elif isinstance(z_eval, float) or isinstance(z_eval, int):
        xlabel = 'halo mass at z=' +str(z_eval)
        halo_mass_list = []
        halo_concentration_list = []
        for halo in realization.halos:
            if halo.is_subhalo and halo.z == z_eval:
                halo_mass_list.append(halo.mass)
                halo_concentration_list.append(halo.c)
    elif isinstance(z_eval, list):
        xlabel = 'halo mass (' + str(z_eval[0])+' < z < '+str(z_eval[1])+')'
        halo_mass_list = []
        halo_concentration_list = []
        for halo in realization.halos:
            if halo.is_subhalo:
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
        log10c_median = np.median(np.log10(halo_concentrations[inds]))
        log10c_std = np.std(np.log10(halo_concentrations[inds]))
        c.append(log10c_median)
        c_error.append(log10c_std)
    log10m = np.log10(x)[0:-1]
    if show_errorbars:
        ax.errorbar(log10m, c, yerr=c_error, fmt='none', color=color)
    ax.plot(log10m, c, color=color, **kwargs_plot)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(r'$\log_{10} c$', fontsize=15)


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
    mass_bound = []
    for halo in realization.halos:
        if halo.is_subhalo:
            masses.append(halo.mass)
            mass_bound.append(halo.bound_mass)

    ax.scatter(masses, mass_bound, color=color, **kwargs_plot)
    ax.plot(masses, masses, color='0.5', lw=2, linestyle=':')
    ax.set_xlim(10**log_mlow, 10**log_mhigh)
    ax.set_xlabel('infall halo mass '+r'$\left[M_{\odot}\right]$', fontsize=15)
    ax.set_ylabel('subhalo mass '+r'$M\left(r < r_{200}\right)$', fontsize=15)
    ax.set_xscale('log')
    ax.set_yscale('log')

def plot_subhalo_mass_functon(realization, log_m_low=6.0, log_m_high=10.0, bound_mass_function=False, nbins=20,
                                  n_bootstrap=10, ax=None, color='k', kwargs_plot={}):
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
        logm = _x[1:] - (_x[1] - _x[0])/2
        ax.errorbar(logm, median, yerr=standard_dev, fmt='none', color=color)
        ax.plot(logm, median, color=color, **kwargs_plot)
        ax.set_ylim(median[-1] / 2, median[0] * 2)
        ax.set_xlabel('infall halo mass ' + r'$\left[M_{\odot}\right]$', fontsize=15)
        ax.set_ylabel(r'$n\left(m\right)$', fontsize=15)
        ax.set_yscale('log')
