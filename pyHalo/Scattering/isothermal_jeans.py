import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.integrate import quad
from pyHalo.Scattering.vdis_nfw import _velocity_dispersion_NFW
from copy import copy
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Halos.lens_cosmo import LensCosmo
from pyHalo.Scattering.cross_sections import VelocityDependentCross

cosmo = Cosmology()

def halo_age(z):

    return cosmo.halo_age(z)

def nfwprofile_density(r, rhos, rs):
    x = r * rs ** -1
    return rhos * (x*(1+x)**2) ** -1

def velocity_dispersion_NFW_slow(r, rho_s, rs):
    G = 4.3e-6  # units kpc and solar mass

    rhos_tilde = lambda x: (x*(1+x)**2) ** -1
    m_tilde = lambda x: np.log(1+x) - x*(1+x)**-1
    integrand_tilde = lambda x: rhos_tilde(x) * m_tilde(x) * x ** -2

    if isinstance(r, float) or isinstance(r, int):

        integral = quad(integrand_tilde, r*rs**-1, 50 * r*rs**-1)[0]
    else:
        integral = []
        for ri in r:
            integral.append(quad(integrand_tilde, ri*rs**-1, 50 * ri*rs**-1)[0])
        integral = np.array(integral)

    return np.sqrt(4*np.pi*G * rs ** 2 * rho_s * integral * rhos_tilde(r*rs**-1)**-1)

def velocity_dispersion_NFW(r, rhos, rs):

    return _velocity_dispersion_NFW(r, rhos, rs)

def energyNFW_slow(rhos, rs, r):

    def _integrand(R):

        vdis = velocity_dispersion_NFW(R, rhos, rs)
        den = nfwprofile_density(R, rhos, rs)

        return 0.5 * den * vdis ** 2 * 4 * np.pi * R ** 2

    if isinstance(r, float) or isinstance(r, int):
        return quad(_integrand, 0, r)[0]
    else:
        integral = []
        for ri in r:
            integral.append(quad(_integrand, 0, ri)[0])
        integral = np.array(integral)
        return integral

def ode_system(x, f):
    z1 = f[1]
    z2 = -2 * x ** -1 * f[1] - np.exp(f[0])
    return [z1, z2]

def integrate_profile(rho0, s0, r_s, r_1, rmax_fac=1.2, rmin_fac=0.01,
                      r_min=None, r_max=None):

    G = 4.3e-6  # units kpc and solar mass
    length_scale = np.sqrt(s0 ** 2 * (4 * np.pi * G * rho0) ** -1)

    if r_max is None:
        x_max = rmax_fac * r_1 / length_scale
    else:
        x_max = r_max/length_scale

    if r_min is None:
        x_min = r_s * rmin_fac / length_scale
    else:
        x_min = r_min/length_scale

    # solve the ODE with initial conditions
    phi_0, phi_prime_0 = 0, 0
    N = 600

    xvalues = np.linspace(x_min, x_max, N)

    res = solve_ivp(ode_system, (x_min, x_max),
                    [phi_0, phi_prime_0], t_eval=xvalues)

    return res['t'] * length_scale, rho0 * np.exp(res.y[0])

def profile_mass(r_iso, rho_iso, rmax):

    m = 0
    ri = 0
    dr = r_iso[1] - r_iso[0]
    idx = 0

    while ri <= rmax:

        m += 4*np.pi*ri**2 * rho_iso[idx] * dr
        idx += 1
        ri = r_iso[idx]
        dr = r_iso[idx+1] - ri

    return m

def profile_density(r, r_iso, rho_iso):

    interp = interp1d(r_iso, rho_iso)

    return interp(r)

def nfwprofile_mass(rhos, rs, rmax):
    x = rmax * rs ** -1
    return 4*np.pi*rhos*rs**3 * (np.log(1+x) - x * (1+x) ** -1)

def compute_k(rhos, cm2_per_gram_times_sigmav, age):

    # cm^2 * solar masses * km / (kpc^3 * gram * sec)
    const = 2.14e-10

    return const * rhos * cm2_per_gram_times_sigmav * age

def compute_r1(rhos, rs, v, cross_section_norm, v_power, t_halo):

    """

    :param rhos: units solar mass / kpc^3
    :param rs: kpc
    :param sigma_v: km/sec
    :param cross_section_class: class cm^2/gram
    :param halo_age: units Gyr
    :return:
    """

    cross_class = VelocityDependentCross(cross_section_norm, v_pow=v_power)
    cross_section_times_v = cross_class(v)
    k = compute_k(rhos, cross_section_times_v, t_halo)
    roots = np.roots([1, 2, 1, -k])
    lam = np.real(np.max(roots[np.where(np.isreal(roots))]))

    return lam * rs

def solve(rhonfw, rsnfw, cross_section_norm, v_power, t_halo, rmin_fac, rmax_fac,
          rho_start, rho_end, s0_start, s0_end, N, plot=False):

    s0nfw = velocity_dispersion_NFW(rsnfw, rhonfw, rsnfw)
    rhocenter = [rhonfw]

    if plot: print('NFW profile velocity dispersion (at rs):', np.round(s0nfw, 2))

    percent = [int(N ** 2 * p) for p in [0.25, 0.5, 0.75]]
    logrhovals = np.linspace(np.log10(rho_start), np.log10(rho_end), N)
    s0vals = np.linspace(s0_start, s0_end, N)
    logrhoarr, s0arr = np.meshgrid(logrhovals, s0vals)

    coords = np.vstack([logrhoarr.ravel(), s0arr.ravel()]).T
    denarr = []
    marr = []
    pcount = 0

    for i in range(0, int(coords.shape[0])):
        if i in percent and plot:
            print(str(100 * percent[pcount] / N ** 2) + '% .... ')
            pcount += 1

        r1 = compute_r1(rhonfw, rsnfw, coords[i, 1], cross_section_norm, v_power, t_halo)
        r_iso, rho_iso = integrate_profile(10 ** coords[i, 0],
                                           coords[i, 1], rsnfw, r1, rmin_fac=rmin_fac,
                                           rmax_fac=rmax_fac)

        mass_nfw, mass_iso = nfwprofile_mass(rhonfw, rsnfw, r1), profile_mass(r_iso, rho_iso, r1)
        nfw_den, sidm_den = nfwprofile_density(r1, rhonfw, rsnfw), profile_density(r1, r_iso, rho_iso)

        mr = mass_nfw / mass_iso
        dr = nfw_den / sidm_den

        mass_pen = np.absolute(mr - 1)
        den_pen = np.absolute(dr - 1)

        marr.append(mass_pen)
        denarr.append(den_pen)

    denarr, marr = np.absolute(np.array(denarr).reshape(N, N)), \
                   np.absolute(np.array(marr).reshape(N, N))

    tot = marr + denarr

    minidx = np.argmin(tot.ravel())
    fit_quality = tot.ravel()[minidx]
    if plot: print('fit: ', tot.ravel()[minidx])
    rho0, s0 = 10 ** coords[minidx, 0], coords[minidx, 1]

    core_density_ratio = rhonfw / rho0

    rhocenter.append(rho0)
    if plot:

        import matplotlib.pyplot as plt

        e1, e2 = logrhovals[0], logrhovals[-1]

        extent = [e1, e2, s0vals[0], s0vals[-1]]
        plt.imshow(np.log10(tot), origin='lower',
                   aspect='auto', cmap='jet', extent=extent);
        plt.scatter(np.log10(rho0), s0, color='w', s=140, marker='x')
        ax = plt.gca()

        ax.annotate('core size (units rs):\n' + str(np.round(core_density_ratio, 2)),
                    xy=(0.6, 0.65), xycoords='axes fraction', fontsize=13)

        text = r'$\sigma = $' + str(cross_section_norm) + ' cm^2 g^-1' + \
                   '\n ' + r'$\sigma \propto v$' + '^' + str(-v_power)

        ax.annotate(text, xy=(0.6, 0.8), xycoords='axes fraction', fontsize=13)
        # plt.colorbar(label=r'$\log_{10}\left(\chi^2\right)$')

        ax.set_xlabel(r'$\log_{10}\left(\rho_0\right)$', fontsize=14)
        ax.set_ylabel(r'$\sigma_v \ \left[\rm{km} \rm{sec^-1} \right]$', fontsize=14)
        plt.show()
        a=input('continue')

    keywords = {'r1': r1, 'r_iso': r_iso, 'rho_iso': rho_iso,
                'rhonfw': rhonfw, 'rsnfw': rsnfw}

    return rho0, s0, core_density_ratio, fit_quality, keywords

def solve_iterative(rhonfw, rsnfw, cross_section_norm, v_power, t_halo,
                    N, plot=False, rmin_fac=0.01, rmax_fac=1.2, tol = 0.002):

    rho0, s0, core_size_unitsrs, fit_quality, keywords = _solve_iterative(
        rhonfw, rsnfw, cross_section_norm, v_power, t_halo, rmin_fac, rmax_fac,
                                                                N, plot=plot, tol=tol,
                                                                s0min_scale=0.4, s0max_scale=1.6,
                                                                rhomin_scale=0.25, rhomax_scale=3.6,
                                                                          )

    # r_iso, rho_iso = keywords['r_iso'], keywords['rho_iso']
    # import matplotlib.pyplot as plt
    # plt.loglog(r_iso, nfwprofile_density(r_iso, rhonfw, rsnfw), color='k')
    # plt.loglog(r_iso, rho_iso, color='r')
    # plt.show()

    if fit_quality > 0.1:
        rho0, s0, core_size_unitsrs, fit_quality, keywords = _solve_iterative(
            rhonfw, rsnfw, cross_section_norm, v_power, t_halo, rmin_fac, rmax_fac,
                                                                    8, plot=plot, tol=tol,
                                                                    s0min_scale=0.2, s0max_scale=2.0,
                                                                    rhomin_scale=0.1, rhomax_scale=4.5)
    if fit_quality > 0.1:
        rho0, s0, core_size_unitsrs, fit_quality = np.nan, np.nan, np.nan, np.nan

    return rho0, s0, core_size_unitsrs, fit_quality, keywords

def _solve_iterative(rhonfw, rsnfw, cross_section_norm, v_power, t_halo,
                     rmin_fac, rmax_fac,
                     N, plot=False, tol = 0.004, do_E = False,
                     s0min_scale=0.5, s0max_scale=1.5,
                     rhomin_scale=0.3, rhomax_scale=3.5):

    fit_quality = 100

    rhomin = rhonfw * rhomin_scale
    rhomax = rhonfw * rhomax_scale
    rho_range = np.log10(rhomax) - np.log10(rhomin)

    s0nfw = velocity_dispersion_NFW(rsnfw, rhonfw, rsnfw)
    s0min, s0max = s0nfw*s0min_scale,  s0nfw*s0max_scale

    s0_range = s0max - s0min

    core_size_unitsrs_last = 1e+8
    fit_quality_last = 1e+9

    iter_count = 1
    N_iter_max = 8

    k = 4
    Nvalues = [N]*k
    iter_step = [1]*k

    for i in range(k,N_iter_max):
        Nvalues.append(int(Nvalues[i-1] + 5 + i + 1))
        iter_step.append(i+2)

    while fit_quality > tol:

        rho0, s0, core_size_unitsrs, fit_quality, keywords = solve(rhonfw, rsnfw, cross_section_norm, v_power,
                                                                   t_halo, rmin_fac, rmax_fac,
                         rhomin, rhomax, s0min, s0max, Nvalues[iter_count-1], plot=plot, do_E=do_E, do_v=False)

        core_size_unitsrs = np.round(core_size_unitsrs, 2)

        new_rho_range = rho_range * iter_step[iter_count-1] ** -1

        rhomin = 10**(np.log10(rho0) - 0.5*new_rho_range)
        rhomax = 10**(np.log10(rho0) + 0.5*new_rho_range)

        new_s0_range = s0_range * iter_step[iter_count] ** -1
        s0min = s0 - 0.5 * new_s0_range
        s0max = s0 + 0.5 * new_s0_range

        if (core_size_unitsrs == core_size_unitsrs_last) and (fit_quality < 0.5*fit_quality_last):
            break

        core_size_unitsrs_last = copy(core_size_unitsrs)
        fit_quality_last = copy(fit_quality)

        if iter_count > len(Nvalues)-1:
            break

        iter_count += 1

        if iter_count >= N_iter_max:
            break

    if fit_quality > 0.1:
        rho0 = np.nan
        s0 = np.nan
        core_size_unitsrs = np.nan

    return rho0, s0, core_size_unitsrs, fit_quality, keywords

