import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.integrate import quad
from pyHalo.Scattering.vdis_nfw import _velocity_dispersion_NFW
from pyHalo.Scattering.Enfw import _energyNFW
from copy import copy
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Halos.lens_cosmo import LensCosmo

class ISONFW(object):

    def __init__(self, riso, rhoiso, rhos_nfw, rs_nfw, rmatch):

        self._iso_rmin = riso[0]
        self._iso_rmax = riso[-1]
        self._rho0 = rhoiso[0]

        self._rhos = rhos_nfw
        self._rs = rs_nfw

        self._rinterpmin = riso[0]

        if rmatch > riso[-1]:
            print('Warning, you are trying to match profiles outside interpolation range. Setting rmatch = riso[-1].')
            rmatch = riso[-1]

        self._rinterpmax = rmatch

        self._rho_inner_interp = interp1d(riso, rhoiso)

    def _eval(self, r):

        if r > self._rinterpmax:
            return self._rhonfw(r)
        elif r<self._rinterpmin:
            return self._rho0
        else:
            return self._rho_inner_interp(r)

    def __call__(self, r):

        if isinstance(r, float) or isinstance(r, int):
            return self._eval(r)
        else:
            out = [self._eval(ri) for ri in r]
            return np.array(out)

    def _rhonfw(self, r):

        x= r * self._rs ** -1

        return self._rhos*(x * (1+x)**2)**-1

    def join_profiles(self, rvals, rho1, rho2, r_join):
        idx = np.argsort(np.absolute(rvals - r_join))[0]
        rho_past1 = rho1[idx:]
        rho_past2 = rho2[idx:]
        idx = np.argsort(np.absolute(rho_past1 - rho_past2))[0]
        return np.append(rho1[0:idx], rho2[idx:])


def halo_age(z):

    cosmo = Cosmology()

    universe_age_today = cosmo.lookback_time(0)

    universe_age_z = universe_age_today - cosmo.lookback_time(z)

    formation_age = 10

    return formation_age - universe_age_z

def cored_profile(r, rhocore, rcore, k=5):
    x = r * rcore ** -1
    return rhocore * (1 + x ** k)

def _interpolating_function(r, r_core, rs):
    arg = (r - r_core) / (rs - r_core)
    if isinstance(r, np.ndarray):

        out = 0.5 * (1 + np.cos(np.pi * arg))
        out[np.where(r < r_core)] = 1
        out[np.where(r > rs)] = 0
        return out
    else:
        if r < r_core:
            return 0
        elif r > r_core:
            return 1
        else:
            return 0.5 * (1 + np.cos(np.pi * arg))


def _hybrid_analytic_point(r, rhocore, rcore, rhos, rmatch):
    f = _interpolating_function(r, rcore, rmatch)
    return (1 - f) * cored_profile(r, rhocore, rcore) + nfwprofile_density(r, rhos, rs) * f


def hybrid_analytic(r, rhocore, rcore, rhos, rmatch):
    f = _interpolating_function(r, rcore, rmatch)

    return f * cored_profile(r, rhocore, rcore) + (1 - f) * nfwprofile_density(r, rhos, rs)

def first_crossing(rvalues, rho_iso, rho_nfw):

    assert len(rvalues) == len(rho_iso)
    assert len(rvalues) == len(rho_nfw)
    diff = rho_iso - rho_nfw
    try:
        diff =np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0] + 1
        return rvalues[diff[0]]
    except:
        idx = np.argsort(np.absolute(diff))
        return rvalues[idx]

def modified_nfwprofile_density(r, rhos, rs, rc, a = 10):

    x = r * rs ** -1
    beta = rc * rs ** -1
    a_inv = a ** -1
    return rhos / (((beta ** a + x ** a) ** a_inv) * (1+x)**2)

def modified_burkert_density(rho0, rs, rc, r):

    x = r * rs ** -1
    beta = rc * rs ** -1

    return rho0 * ((beta + x)*(1 + x**2)) ** -1

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

def energyNFW(rhos, rs, r):
    # numerical fudge factor
    return _energyNFW(r, rhos, rs) * 1.045

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

def integrate_profile(rho0, s0, r_s, r_1, rmax_fac=1.2):

    G = 4.3e-6  # units kpc and solar mass
    length_scale = np.sqrt(s0 ** 2 * (4 * np.pi * G * rho0) ** -1)

    x_max = rmax_fac * r_1 * length_scale ** -1

    # solve the ODE with initial conditions
    phi_0, phi_prime_0 = 0, 0
    N = 600

    x_min = r_s * 0.01 * length_scale ** -1
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

    try:
        return interp(r)
    except:
        index = np.where(r_iso>r)[0]
        return rho_iso[index]

def nfwprofile_mass(rhos, rs, rmax):
    x = rmax * rs ** -1
    return 4*np.pi*rhos*rs**3 * (np.log(1+x) - x * (1+x) ** -1)

def compute_r1(rhos, rs, sigma_v, cross_section_class):

    tscale = 10 # Gyr
    cross_section_times_v = cross_section_class(sigma_v)
    k = 0.52 * (rhos * 10**-8) * cross_section_times_v * (tscale * 0.1)
    roots = np.roots([1, 2, 1, -k])
    lam = np.real(np.max(roots[np.where(np.isreal(roots))]))

    return lam * rs

def solve(rhonfw, rsnfw, cross_section_class,
          rho_start, rho_end, s0_start, s0_end, N, plot=False, do_E=False, do_v=False):

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
    earr = []
    varr = []
    pcount = 0
    cfirstcross = []

    for i in range(0, int(coords.shape[0])):
        if i in percent and plot:
            print(str(100 * percent[pcount] / N ** 2) + '% .... ')
            pcount += 1

        r1 = compute_r1(rhonfw, rsnfw, coords[i, 1], cross_section_class)
        r_iso, rho_iso = integrate_profile(10 ** coords[i, 0], coords[i, 1], rsnfw, r1, rmax_fac=2)

        mass_nfw, mass_iso = nfwprofile_mass(rhonfw, rsnfw, r1), profile_mass(r_iso, rho_iso, r1)
        nfw_den, sidm_den = nfwprofile_density(r1, rhonfw, rsnfw), profile_density(r1, r_iso, rho_iso)

        mr = mass_nfw / mass_iso
        dr = nfw_den / sidm_den

        mass_pen = np.absolute(mr - 1)
        den_pen = np.absolute(dr - 1)

        marr.append(mass_pen)
        denarr.append(den_pen)
        cfirstcross.append(first_crossing(r_iso, rho_iso, nfwprofile_density(r_iso, rhonfw, rsnfw)))

        if do_E:
            er = energyNFW(rhonfw, rsnfw, r1) * (0.5 * coords[i, 1] ** 2 * mass_iso) ** -1
            er_pen = np.absolute(er - 1)
            earr.append(er_pen)

        if do_v:
            vpen = coords[i, 1] * velocity_dispersion_NFW(r1, rhonfw, rsnfw) ** -1
            varr.append(np.absolute(vpen - 1))

    denarr, marr = np.absolute(np.array(denarr).reshape(N, N)), \
                   np.absolute(np.array(marr).reshape(N, N))


    if do_E:
        earr = np.absolute(np.array(earr).reshape(N, N))
        tot = marr + earr
    elif do_v:
        varr = np.absolute(np.array(varr).reshape(N, N))
        tot = marr + varr
    else:

        tot = marr + denarr

    minidx = np.argmin(tot.ravel())
    fit_quality = tot.ravel()[minidx]
    if plot: print('fit: ', tot.ravel()[minidx])
    rho0, s0 = 10 ** coords[minidx, 0], coords[minidx, 1]

    #core_first_crossing = cfirstcross[minidx]
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

        if cross_section_class.has_v_dep:
            text = r'$\sigma = $' + str(cross_section_class._cross) + ' cm^2 g^-1' + \
                   '\n ' + r'$\sigma \propto v$' + '^' + str(-cross_section_class.v_pow)
        else:
            text = r'$\sigma = $' + str(cross_section_class._cross) + ' cm^2 g^-1\n' + \
                   '(v-independent)'
        ax.annotate(text, xy=(0.6, 0.8), xycoords='axes fraction', fontsize=13)
        # plt.colorbar(label=r'$\log_{10}\left(\chi^2\right)$')

        ax.set_xlabel(r'$\log_{10}\left(\rho_0\right)$', fontsize=14)
        ax.set_ylabel(r'$\sigma_v \ \left[\rm{km} \rm{sec^-1} \right]$', fontsize=14)
        plt.show()
        a=input('continue')

    return rho0, s0, core_density_ratio, fit_quality

def solve_iterative(rhonfw, rsnfw, cross_section_class, N, plot=False, tol = 0.002):

    rho0, s0, core_size_unitsrs, fit_quality = _solve_iterative(rhonfw, rsnfw, cross_section_class,
                                                                N, plot=plot, tol=tol)

    #if fit_quality > 0.1:
    #    print('using energy boundary conditions')
    #    rho0, s0, core_size_unitsrs, fit_quality = _solve_iterative(rhonfw, rsnfw, cross_section_class, N, do_E=True)

    return rho0, s0, core_size_unitsrs, fit_quality

def _solve_iterative(rhonfw, rsnfw, cross_section_class, N, plot=False, tol = 0.002, do_E = False):

    fit_quality = 100

    rhomin = rhonfw * 0.4
    rhomax = rhonfw * 3
    rho_range = np.log10(rhomax) - np.log10(rhomin)

    s0nfw = velocity_dispersion_NFW(rsnfw, rhonfw, rsnfw)
    s0min, s0max = s0nfw*0.5,  s0nfw*1.5

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

        rho0, s0, core_size_unitsrs, fit_quality = solve(rhonfw, rsnfw, cross_section_class,
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

    return rho0, s0, core_size_unitsrs, fit_quality

def solve_mass_range(masses, common_redshift, cross_class):

    logrho0 = []
    prof = LensCosmo(z_lens=0.5, z_source=3)
    fitquality = []

    for mi in masses:
        rhos, rs, _ = prof.NFW_params_physical_fromM(mi, common_redshift)
        rho0, _, _, fit_qual = solve_iterative(rhos, rs, cross_class, 5)
        logrho0.append(np.log10(rho0))
        fitquality.append(fit_qual)

    cut = np.where(np.array(fitquality) < 0.05)[0]

    return np.array(masses)[cut], np.array(logrho0)[cut], np.array(fitquality)[cut]


def solve_z_range(mass, redshifts, cross_class):

    logrho0 = []
    prof = LensCosmo(z_lens=0.5, z_source=3)
    fitquality = []

    for zi in redshifts:
        rhos, rs, _ = prof.NFW_params_physical_fromM(mass, zi)
        rho0, _, _, fit_qual = solve_iterative(rhos, rs, cross_class, 5)
        logrho0.append(np.log10(rho0))
        fitquality.append(fit_qual)

    cut = np.where(np.array(fitquality) < 0.05)

    return np.array(logrho0)[cut], np.array(fitquality)[cut]

def solve_cross_range(mass, z, cross_classes):
    logrho0 = []
    prof = LensCosmo(z_lens=0.5, z_source=3)
    fitquality = []
    for cross_class in cross_classes:
        rhos, rs, _ = prof.NFW_params_physical_fromM(mass, z)
        rho0, _, _, fit_qual = solve_iterative(rhos, rs, cross_class, 5)
        logrho0.append(np.log10(rho0))
        fitquality.append(fit_qual)

    cut = np.where(np.array(fitquality) < 0.05)

    return np.array(logrho0)[cut], np.array(fitquality)[cut]
