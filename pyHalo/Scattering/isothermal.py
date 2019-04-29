import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, newton, minimize
from scipy.integrate import quad
from copy import copy
from mpmath import polylog

def zeta(cross, tscale = 10):
    return cross * tscale

def compute_r1(rhos, rs, sigma, cross):

    # 1 cm^2/gram km/sec = 2.3165e-10 kpc^3 / solar mass / Gyr

    v_avg = 4 * 3.14**-0.5 * zeta(cross) * sigma

    const = 2.316e-10
    k = v_avg * rhos * const

    roots = np.roots([1, 2, 1, -k])
    lam = np.real(np.max(roots[np.where(np.isreal(roots))]))

    return lam * rs

def ode_system(x, f):
    z1 = f[1]
    z2 = -2 * x ** -1 * f[1] - np.exp(f[0])
    return [z1, z2]


def integrate_profile(rho0, s0, r_s, r_1, rmax_fac = 1.2,
                      rvalues = None):
    G = 4.3e-6  # units kpc and solar mass

    length_scale = np.sqrt(s0 ** 2 * (4 * np.pi * G * rho0) ** -1)

    x_max = rmax_fac * r_1 * length_scale ** -1

    # solve the ODE with initial conditions
    phi_0, phi_prime_0 = 0, 0
    N = 3000

    if rvalues is None:
        x_min = r_s * 0.01 * length_scale ** -1
        xvalues = np.linspace(x_min, x_max, N)
    else:
        xvalues = rvalues * length_scale
        x_min, x_max = xvalues[0], xvalues[-1]

    res = solve_ivp(ode_system, (x_min, x_max),
                    [phi_0, phi_prime_0], t_eval=xvalues)

    return res['t'] * length_scale, rho0 * np.exp(res.y[0])


def nfwmass(rho_s, rs, r1):
    c = r1 * rs ** -1
    cfac = np.log(1 + c) - c * (1 + c) ** -1
    return 4 * np.pi * rs ** 3 * rho_s * cfac


def profile_mass(rvalues, rhovalues, rmax):
    mass = 1
    i = 0
    ri = rvalues[i]
    rnext = rvalues[i + 1]
    dri = rnext - ri

    while rnext < rmax:
        mass += 4 * np.pi * ri ** 2 * rhovalues[i] * dri
        i += 1
        ri = rvalues[i]
        rnext = rvalues[i + 1]
        dri = rnext - ri

    return mass

def density_sidm(rsidm, rhosidm, r):

    try:
        interp_sidm = interp1d(np.log10(rsidm), np.log10(rhosidm))
        logrho = interp_sidm(np.log10(r))
    except:
        #print(r)
        index = np.where(rsidm < r)[0]
        logrho = np.log10(rhosidm[index])

    return 10**logrho

def density_nfw(rhos, rs, r):
    x = r * rs ** -1
    fac = x * (1+x)**2
    return rhos * fac**-1

def mass_ratio(r1, nfwrho, nfwrs, rsidm, sidmrho):

    mnfw = nfwmass(nfwrho, nfwrs, r1)
    msidm = profile_mass(rsidm, sidmrho, r1)

    return np.log10(mnfw) - np.log10(msidm)

def fit_sidm(rho0_proposal, vdis_proposal,
             nfw_rhos, nfw_rs, sidm_cross):

    r1 = compute_r1(nfw_rhos, nfw_rs, vdis_proposal, sidm_cross)
    r_iso, rho_iso = integrate_profile(rho0_proposal, vdis_proposal,
                                       nfw_rs, r1)

    massnfw = nfwmass(nfw_rhos, nfw_rs, r1)
    sidmmass = profile_mass(r_iso, rho_iso, r1)

    nfwden = density_nfw(nfw_rhos, nfw_rs, r1)
    sidmdensity = density_sidm(r_iso, rho_iso, r1)

    density_ratio = sidmdensity / nfwden
    mass_ratio = massnfw / sidmmass

    return density_ratio, mass_ratio

def minimize_grid(rhonfw, rsnfw, cross):

        N = 20
        logrhovals = np.linspace(6, 8.4, N)
        s0vals = np.linspace(0.5, 10, N)
        logrhoarr, s0arr = np.meshgrid(logrhovals, s0vals)

        coords = np.vstack([logrhoarr.ravel(), s0arr.ravel()]).T
        denarr = []
        marr = []
        earr = []

        for i in range(0, int(coords.shape[0])):

            mass_ratio, density_ratio, energyratio = \
                all_ratios(10**coords[i,0], coords[i,1], rhonfw,
                           rsnfw, cross)
            marr.append(mass_ratio)
            denarr.append(density_ratio)
            earr.append(energyratio)

        denarr, marr, earr = np.absolute(np.array(denarr).reshape(N, N) - 1), \
                             np.absolute(np.array(marr).reshape(N,N) - 1), \
                             np.absolute(np.array(earr).reshape(N, N) - 1)


        tot = marr + earr
        minidx = np.argmin(tot.ravel())
        rho0, s0 = 10**coords[minidx, 0], coords[minidx,1]

        print('solution:', np.log10(rho0), s0)
        print('density ratio:', denarr.ravel()[minidx])
        print('mass ratio:', marr.ravel()[minidx])
        print('energy ratio:', earr.ravel()[minidx])
        print('core size:', rhonfw * rho0 ** -1)

        plt.imshow(np.log10(tot), origin='lower', extent = [logrhovals[0], logrhovals[-1], s0vals[0], s0vals[-1]],
                   aspect='auto'); plt.colorbar()
        plt.show()

def makeplot(rho0, s0, rhonfw, rsnfw, cross):

    r1 = compute_r1(rhonfw, rsnfw, s0, cross)
    rvals = np.linspace(rsnfw * 0.1, 2 * rsnfw, 5000)
    r, rho = integrate_profile(rho0, s0, rsnfw, r1, rmax_fac=20, rvalues=rvals)

    nfwrho = density_nfw(rhonfw, rsnfw, rvals)

    plt.loglog(rvals, rho, color='r')
    plt.loglog(rvals, nfwrho, color='k')
    plt.axvline(r1)
    plt.show()

    massnfw = nfwmass(rhonfw, rsnfw, r1)
    sidmmass = profile_mass(rvals, rho, r1)

    nfwden = density_nfw(rhonfw, rsnfw, r1)
    sidmdensity = density_sidm(rvals, rho, r1)

    density_ratio = sidmdensity / nfwden
    mass_ratio = massnfw / sidmmass
    print(mass_ratio, density_ratio)

    energysidm = 0.5 * sidmmass * s0**2
    energynfw = energyNFW(rhonfw, rsnfw, r1)
    print(energysidm / energynfw)

def all_ratios(rho0, s0, rhonfw, rsnfw, cross):
    r1 = compute_r1(rhonfw, rsnfw, s0, cross)
    rvals = np.linspace(rsnfw * 0.01, 60 * rsnfw, 5000)
    r, rho = integrate_profile(rho0, s0, rsnfw, r1, rmax_fac=20, rvalues=rvals)

    massnfw = nfwmass(rhonfw, rsnfw, r1)
    sidmmass = profile_mass(rvals, rho, r1)

    nfwden = density_nfw(rhonfw, rsnfw, r1)
    sidmdensity = density_sidm(rvals, rho, r1)

    density_ratio = sidmdensity / nfwden
    mass_ratio = massnfw / sidmmass

    energysidm = 0.5 * sidmmass * s0 ** 2
    energynfw = energyNFW(rhonfw, rsnfw, r1)
    energyratio = energynfw * energysidm ** -1

    return mass_ratio, density_ratio, energyratio

def velocity_dispersion_NFW_exact(r, rho_s, rs):

    G = 4.3e-6  # units kpc and solar mass

    if isinstance(r, float) or isinstance(r, int):
        x = r * rs ** -1
        vdis_squared = G * (2 * rho_s * np.pi * rs ** 3 * (x * (-1 + x * (-9 - 7 * x + np.pi ** 2 * (1 + x) ** 2)) -
          x ** 2 * (1 + x) ** 2 * np.log(x) +
          (1 + x) * np.log(1 + x) * (1 + x * (-3 + (-5 + x) * x) + 3 * x ** 2 * (1 + x) * np.log(1 + x)) +
          6 * x ** 2 * (1 + x) ** 2 * polylog(2, -x))) / x

    else:
        shape0 = np.shape(r)
        vdis_squared = np.zeros_like(r).ravel()
        for i, ri in enumerate(r):
            x = ri * rs**-1
            vdis_squared[i] = G * (2 * rho_s * np.pi * rs ** 3 * (x * (-1 + x * (-9 - 7 * x + np.pi ** 2 * (1 + x) ** 2)) -
          x ** 2 * (1 + x) ** 2 * np.log(x) +
          (1 + x) * np.log(1 + x) * (1 + x * (-3 + (-5 + x) * x) + 3 * x ** 2 * (1 + x) * np.log(1 + x)) +
          6 * x ** 2 * (1 + x) ** 2 * polylog(2, -x))) / x

        vdis_squared = vdis_squared.reshape(shape0)

    return vdis_squared ** 0.5

def velocity_dispersion_NFW(r, rho_s, rs):

    G = 4.3e-6  # units kpc and solar mass

    rho_at_r = density_nfw(rho_s, rs, r)

    def _integrand(R):

        mass_nfw = nfwmass(rho_s, rs, R)
        den_nfw = density_nfw(rho_s, rs, R)
        X = R * rs ** -1
        return den_nfw * mass_nfw * X ** -2

    if isinstance(r, float) or isinstance(r, int):

        integral = quad(_integrand, r, 10*rs)[0]
    else:
        integral = []
        for ri in r:
            integral.append(quad(_integrand, ri, 100*rs)[0])
        integral = np.array(integral)

    return np.sqrt(G * integral * rho_at_r ** -1)

def energyNFW(rhos, rs, r):

    def _integrand(R):

        vdis = velocity_dispersion_NFW(R, rhos, rs)
        den = density_nfw(rhos, rs, R)

        return 0.5 * den * vdis ** 2 * 4 * np.pi * R ** 2

    if isinstance(r, float) or isinstance(r, int):
        return quad(_integrand, 0, r)[0]
    else:
        integral = []
        for ri in r:
            integral.append(quad(_integrand, 0, ri)[0])
        integral = np.array(integral)
        return integral

def test2():

    rho0 = 3*10**7
    s0 = 5.4

    rhonfw = 8.5 * 10 ** 6
    rsnfw = 0.85
    cross = 5

    makeplot(rho0, s0, rhonfw, rsnfw, cross)

    solution = solve(rhonfw, rsnfw, cross)

    makeplot(solution[0], solution[1], rhonfw, rsnfw, cross)

def solve_old(rhonfw, rsnfw, cross):

    def _penalty_function(args):
        logrho0, s0 = args[0], args[1]
        rho0, s0 = 10**logrho0, 10**s0

        massratio, densityratio, energyratio = \
            all_ratios(rho0, s0, rhonfw, rsnfw, cross)

        p1 = np.absolute(energyratio - 1) * 0.01 ** -1
        p2 = np.absolute(massratio - 1) * 0.01 ** -1
        return p1 + p2

    rhoinit = 3 * 10 ** 7
    s0init = 5
    x0 = np.array([np.log10(rhoinit), np.log10(s0init)])
    solve_method = 'Powell'
    opt = minimize(_penalty_function, x0 = x0, method=solve_method)
    xout = 10**opt['x']
    print(opt)
    #print(xout)
    #print(rhonfw * xout[0] ** -1)
    return xout


def getrangesrho(rhocen, N, step):
    return np.linspace(np.log10(rhocen) - step, np.log10(rhocen) + step, N)


def getrangess0(s0cen, N, step):
    if s0cen - step < 0.5:
        return np.linspace(0.5, s0cen + step, N)
    return np.linspace(s0cen - step, s0cen + step, N)


def solve(rhonfw, rsnfw, cross):
    N = [40, 30]
    step = [3, 1]
    s0step = [2, 1.5]
    s0start = [velocity_dispersion_NFW(rsnfw, rhonfw, rsnfw)]
    rhocenter = [rhonfw]

    for count in range(0, 2):

        if rhocenter[count] - step > 0:
            logrhovals = np.linspace(np.log10(rhocenter[count]) - step[count],
                                     np.log10(rhocenter[count]) + step[count], N[count])
        else:
            logrhovals = np.linspace(1,
                                     np.log10(rhocenter[count]) + step[count], N[count])

        s0vals = np.linspace(s0start[count]*s0step[0]**-1, s0start[count]*s0step[0], N[count])
        logrhoarr, s0arr = np.meshgrid(logrhovals, s0vals)

        coords = np.vstack([logrhoarr.ravel(), s0arr.ravel()]).T
        denarr = []
        marr = []
        earr = []

        for i in range(0, int(coords.shape[0])):
            mass_ratio, density_ratio, energyratio = \
                all_ratios(10 ** coords[i, 0], coords[i, 1], rhonfw,
                           rsnfw, cross)
            marr.append(mass_ratio)
            denarr.append(density_ratio)
            earr.append(energyratio)

        denarr, marr, earr = np.absolute(np.array(denarr).reshape(N[count], N[count]) - 1), \
                             np.absolute(np.array(marr).reshape(N[count], N[count]) - 1), \
                             np.absolute(np.array(earr).reshape(N[count], N[count]) - 1)

        #tot = marr + earr
        tot = marr + denarr
        minidx = np.argmin(tot.ravel())
        rho0, s0 = 10 ** coords[minidx, 0], coords[minidx, 1]
        rhocenter = 10 ** coords[minidx, 0]
        s0start.append(coords[minidx, 1])

        #plt.imshow(np.log10(tot), origin='lower',
        #           extent=[logrhovals[0] / np.log10(rhonfw), logrhovals[-1] / np.log10(rhonfw), s0vals[0], s0vals[-1]],
        #           aspect='auto', cmap='seismic');
        #plt.colorbar()
        #plt.show()

    return rho0, s0


if False:
    logrho = np.log10(8*10**6)
    logrs = np.log10(0.85)

    from pyHalo.Halos.cosmo_profiles import CosmoMassProfiles
    prof = CosmoMassProfiles(z_lens=0.5, z_source=3)
    M = [10**6, 10**7, 10**8, 10**9, 10**10]

    rc = []
    rho0rc = []
    for mi in M:
        c = prof.NFW_concentration(mi, 0.5, scatter=False)
        rho_s, rs, _ = prof.NFW_params_physical(mi, c, 0.5)
        #rho_s = 10**logrho
        #rs = 10**logrs
        out = solve(rho_s, rs, 5)

        rho0, s0 = out[0], out[1]
        print(rho_s / rho0, s0)
        #print(all_ratios(rho0, s0, rho_s, rs, 5))
        rc.append(rho_s / rho0)
        rho0rc.append(rho0 * rc[-1])
    print(rho0rc)
    plt.plot(np.log10(M), rc); plt.show()

if False:
    rhos_min = 7*10**6
    rhos_max = 2*10**8

    rs_min = 0.1
    rs_max = 5

    cross_min = 2.6
    cross_max = 10

    N = 6
    cross_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    logrhos_values = np.linspace(np.log10(rhos_min), np.log10(rhos_max), N)
    logrs_values = np.linspace(np.log10(rs_min), np.log10(rs_max), N)

    def solve_array(logrhovals, logrsvals, c):
        sols = np.zeros((N, N))
        for i, logrho in enumerate(logrhovals):
            for j, logrs in enumerate(logrsvals):
                out = solve(10 ** logrho, 10 ** logrs, c)
                sols[i, j] = out[0]
        return sols

    import sys
    #index = 1
    index = int(sys.argv[1])
    cross_section = cross_values[index]
    fname = 'rho_' + str(int(np.round(cross_section * 10))) + '.txt'

    grid = solve_array(logrhos_values, logrs_values, cross_section)

    np.savetxt(fname, X = grid, fmt='%2.2e')

