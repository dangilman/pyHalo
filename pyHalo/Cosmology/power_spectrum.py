"""
The variance of the linear density field, sigma(M, z), computed from a tabulated linear matter power spectrum.

Two window functions are implemented:

TOPHAT: the real-space top-hat, W(x) = 3 (sin x - x cos x) / x^3 with x = kR. This is the window colossus uses, so a
LCDM power spectrum reproduces the Sheth-Tormen mass function and the Ludlow et al. (2016) concentration-mass relation
that pyHalo evaluates through colossus.

BENSON2026: the "extended ETHOS" window function of Benson, Nadler, Du & Gluscevic (2026), arXiv:2606.12137, their
Equations 21-24:

    W(k, R) = 1                                         if kR / c_W(k) < x_min
              [1 + (kR / c_W(k) - x_min)^beta(k)]^-1     otherwise

with c_W(k) = c_W0 * c_W1^(n(k) - n0) and beta(k) = beta0 * beta1^(n(k) - n0). n(k) is the logarithmic slope of the
power spectrum smoothed with a Gaussian kernel in ln k. Three conventions not stated in the paper were confirmed by
A. Benson (priv. comm.) and are followed here (see notebooks/calibrate_benson_hmf.ipynb):
    1) all logs in Equation 24 are natural, and kwidth is the width of the Gaussian in ln k;
    2) n is evaluated at the wavenumber being integrated over, so the window is W(k, R) rather than W(kR);
    3) n(k) is computed by differentiating the Gaussian kernel analytically.

Units follow pyHalo: k in 1/Mpc, P(k) in Mpc^3, masses in M_sun (no factors of h). The Lagrangian radius of a halo of
mass M is R = (3 M / (4 pi rho_m0))^(1/3), with rho_m0 the comoving matter density, for both windows.
"""
import warnings
import numpy as np
from scipy.interpolate import CubicSpline

__all__ = ['LinearPowerSpectrum', 'DELTA_COLLAPSE', 'BENSON2026_WINDOW_PARAMS']

# linear collapse threshold; the value colossus uses for the sheth99 mass function and the ludlow16 model
DELTA_COLLAPSE = 1.68647

# Benson et al. (2026) window function parameters, from the posteriors in their Figure 11
BENSON2026_WINDOW_PARAMS = {'cW0': 3.547, 'beta0': 4.475, 'cW1': 0.786, 'beta1': 1.394,
                            'x_min': 0.004, 'kwidth': 0.686, 'n0': -2.6}

_FILTERS = ['TOPHAT', 'BENSON2026']


class LinearPowerSpectrum(object):
    """
    Computes sigma(M, z) and dln(sigma)/dln(M) from a tabulated linear matter power spectrum at z = 0, for use by the
    mass function and concentration-mass relation classes built from a power spectrum.

    sigma(M) is tabulated on a grid in log10(M) the first time a window function is requested, and splined; the growth
    factor is tabulated from colossus at construction. The instance keeps no reference to the cosmology after
    construction, and copy.deepcopy returns the same instance (it is not modified after construction, except for the
    sigma tables it caches), so it can be passed in kwargs that pyHalo deep-copies at every lens plane.
    """

    def __init__(self, k, pk, cosmology, sigma8=None, little_h_units=False, log10m_min=2.0, log10m_max=17.0,
                 n_mass_per_dex=20, k_min=1e-6, k_max=1e8, n_k=1400, kwargs_benson_window=None):
        """

        :param k: wavenumbers at which the power spectrum is tabulated [1/Mpc, or h/Mpc if little_h_units is True]
        :param pk: the linear matter power spectrum at z = 0 [Mpc^3, or (Mpc/h)^3 if little_h_units is True]; must be
        positive
        :param cosmology: an instance of pyHalo's Cosmology class; sets the matter density and the growth factor
        :param sigma8: if not None, the power spectrum is rescaled so that sigma (top-hat window, R = 8 Mpc/h) equals
        this value; if None, the power spectrum is used with the amplitude it is given
        :param little_h_units: bool; if True, k and pk are in units of h/Mpc and (Mpc/h)^3
        :param log10m_min: log10 of the minimum halo mass [M_sun] at which sigma is tabulated
        :param log10m_max: log10 of the maximum halo mass [M_sun] at which sigma is tabulated
        :param n_mass_per_dex: number of points per decade in halo mass in the sigma table
        :param k_min: minimum wavenumber of the internal integration grid [1/Mpc]
        :param k_max: maximum wavenumber of the internal integration grid [1/Mpc]
        :param n_k: number of points in the internal integration grid, uniformly spaced in ln k. The power spectrum is
        interpolated onto this grid in log-log space, and extrapolated beyond the input table with power laws
        that match the logarithmic slope at each end of the table
        :param kwargs_benson_window: keyword arguments that replace entries of BENSON2026_WINDOW_PARAMS
        """
        k = np.asarray(k, dtype=float)
        pk = np.asarray(pk, dtype=float)
        if k.ndim != 1 or k.shape != pk.shape or len(k) < 4:
            raise ValueError('k and pk must be one-dimensional arrays of the same length (at least 4 points)')
        if np.any(np.diff(k) <= 0):
            raise ValueError('k must be strictly increasing')
        if np.any(pk <= 0) or not np.all(np.isfinite(pk)):
            raise ValueError('the power spectrum must be positive and finite at every tabulated wavenumber')
        if log10m_min >= log10m_max:
            raise ValueError('log10m_min must be less than log10m_max')

        colossus = cosmology.colossus
        self.h = float(cosmology.h)
        self.Om0 = float(colossus.Om0)
        self.Ode0 = float(colossus.Ode0)
        # comoving matter density in M_sun / Mpc^3 (colossus returns M_sun h^2 / kpc^3)
        self.rho_m0 = float(colossus.rho_m(0.0)) * 1e9 * self.h ** 2
        self._astropy_Om0 = float(cosmology.astropy.Om0)

        if little_h_units:
            k = k * self.h
            pk = pk / self.h ** 3
        self._k_table_max = k[-1]

        self._lnk = np.linspace(np.log(k_min), np.log(k_max), n_k)
        self._k = np.exp(self._lnk)
        self._pk = self._interpolate_power_spectrum(k, pk, self._k)
        if k[0] > 1e-3:
            warnings.warn('the power spectrum table starts at k = %.2e / Mpc; it is extrapolated to smaller k with a '
                          'power law' % k[0])
        if sigma8 is not None:
            r8 = 8.0 / self.h
            w = self._tophat(self._k * r8)
            sigma8_input = np.sqrt(np.trapezoid(self._delta2(self._pk) * w ** 2, self._lnk))
            self._pk = self._pk * (sigma8 / sigma8_input) ** 2
        self.sigma8_tophat = self._sigma_exact(np.array([4 * np.pi / 3 * self.rho_m0 * (8.0 / self.h) ** 3]),
                                               'TOPHAT')[0]

        # sigma is computed directly at masses where the input table extends to k R >= 10; below this mass it relies
        # on the extrapolation of the power spectrum beyond the table
        self._m_resolved = 4 * np.pi / 3 * self.rho_m0 * (10.0 / self._k_table_max) ** 3
        self._warned_extrapolation = False

        self._log10m_grid = np.linspace(log10m_min, log10m_max, int(round((log10m_max - log10m_min) *
                                                                          n_mass_per_dex)) + 1)
        self._kwargs_benson_window = dict(BENSON2026_WINDOW_PARAMS)
        if kwargs_benson_window is not None:
            for key in kwargs_benson_window.keys():
                if key not in BENSON2026_WINDOW_PARAMS:
                    raise ValueError('unrecognized Benson window parameter ' + str(key))
            self._kwargs_benson_window.update(kwargs_benson_window)
        self._sigma_tables = {}

        # growth factor, normalized to one at z = 0, tabulated over the redshift range colossus supports
        self.z_min = float(colossus.z_min)
        self.z_max = float(colossus.z_max)
        z = np.expm1(np.linspace(np.log1p(self.z_min), np.log1p(self.z_max), 4000))
        z[0], z[-1] = self.z_min, self.z_max
        self._log1pz_growth = np.log1p(z)
        self._ln_growth = np.log(colossus.growthFactor(z))

    def __deepcopy__(self, memo):
        return self

    def power_spectrum(self, k):
        """
        The linear power spectrum at z = 0, as interpolated (and, if requested, renormalized) on the internal grid
        :param k: wavenumber [1/Mpc]
        :return: P(k) [Mpc^3]
        """
        return np.exp(np.interp(np.log(k), self._lnk, np.log(self._pk)))

    def growth_factor(self, z):
        """
        The linear growth factor normalized to one at z = 0, computed by colossus for the cosmology passed to the
        constructor
        :param z: redshift (number or array)
        :return: D(z)
        """
        u = np.log1p(np.asarray(z, dtype=float))
        if np.any(u < self._log1pz_growth[0]) or np.any(u > self._log1pz_growth[-1]):
            raise ValueError('redshift outside the range of the growth factor table [%.3f, %.1f]' %
                             (self.z_min, self.z_max))
        d = np.exp(np.interp(u, self._log1pz_growth, self._ln_growth))
        if np.ndim(z) == 0:
            return float(d)
        return d

    def lagrangian_radius(self, m):
        """
        :param m: halo mass [M_sun]
        :return: the comoving radius enclosing mass m at the mean matter density [Mpc]
        """
        return (3 * np.asarray(m, dtype=float) / (4 * np.pi * self.rho_m0)) ** (1.0 / 3)

    def sigma(self, m, z=0.0, filter='TOPHAT'):
        """
        The rms variance of the linear density field
        :param m: halo mass [M_sun] (number or array)
        :param z: redshift (number, or array with the same shape as m)
        :param filter: the window function; TOPHAT or BENSON2026
        :return: sigma(m, z)
        """
        log10m = self._check_mass(m)
        table = self._sigma_table(filter)
        s = np.exp(table[0](log10m)) * self.growth_factor(z)
        if np.ndim(m) == 0 and np.ndim(z) == 0:
            return float(s)
        return s

    def dlnsigma_dlnm(self, m, filter='TOPHAT'):
        """
        The logarithmic derivative of sigma with respect to halo mass; independent of redshift
        :param m: halo mass [M_sun] (number or array)
        :param filter: the window function; TOPHAT or BENSON2026
        :return: dln(sigma) / dln(m)
        """
        log10m = self._check_mass(m)
        table = self._sigma_table(filter)
        d = table[1](log10m)
        if np.ndim(m) == 0:
            return float(d)
        return d

    def peak_height(self, m, z, filter='TOPHAT'):
        """
        :param m: halo mass [M_sun]
        :param z: redshift
        :param filter: the window function; TOPHAT or BENSON2026
        :return: delta_c / sigma(m, z)
        """
        return DELTA_COLLAPSE / self.sigma(m, z, filter)

    def check_cosmology(self, cosmology):
        """
        Raises an exception if the cosmology differs from the one used to construct this class
        :param cosmology: an instance of pyHalo's Cosmology class
        """
        if abs(cosmology.h / self.h - 1) > 1e-6 or abs(cosmology.astropy.Om0 / self._astropy_Om0 - 1) > 1e-6:
            raise ValueError('the cosmology (h = %.4f, Om0 = %.4f) differs from the one used to set up the power '
                             'spectrum (h = %.4f, Om0 = %.4f)' % (cosmology.h, cosmology.astropy.Om0, self.h,
                                                                   self._astropy_Om0))

    def _check_mass(self, m):
        """
        :param m: halo mass [M_sun]
        :return: log10(m), after checking that it falls inside the sigma table
        """
        m = np.asarray(m, dtype=float)
        if np.any(m <= 0):
            raise ValueError('halo masses must be positive')
        log10m = np.log10(m)
        if np.any(log10m < self._log10m_grid[0] - 1e-10) or np.any(log10m > self._log10m_grid[-1] + 1e-10):
            raise ValueError('halo mass outside the range of the sigma table, log10(M) in [%.2f, %.2f]; set '
                             'log10m_min and log10m_max when creating the LinearPowerSpectrum' %
                             (self._log10m_grid[0], self._log10m_grid[-1]))
        if not self._warned_extrapolation and np.min(m) < self._m_resolved:
            warnings.warn('sigma(M) below M = %.2e M_sun depends on the extrapolation of the power spectrum beyond '
                          'the largest tabulated wavenumber, k = %.2e / Mpc' % (self._m_resolved, self._k_table_max))
            self._warned_extrapolation = True
        return log10m

    def _sigma_table(self, filter):
        """
        Returns (and caches) splines of ln(sigma) and dln(sigma)/dln(m) in log10(m) at z = 0
        :param filter: the window function; TOPHAT or BENSON2026
        """
        if filter not in self._sigma_tables:
            if filter not in _FILTERS:
                raise ValueError('filter must be one of ' + str(_FILTERS) + ', not ' + str(filter))
            m = 10 ** self._log10m_grid
            ln_sigma = np.log(self._sigma_exact(m, filter))
            spline_ln_sigma = CubicSpline(self._log10m_grid, ln_sigma)
            # as in the calibration of Benson et al., the slope is the derivative of the spline of ln(sigma)
            dlnsigma_dlnm = spline_ln_sigma.derivative()(self._log10m_grid) / np.log(10)
            self._sigma_tables[filter] = (spline_ln_sigma, CubicSpline(self._log10m_grid, dlnsigma_dlnm))
        return self._sigma_tables[filter]

    def _sigma_exact(self, m, filter):
        """
        Integrates the power spectrum with the window function
        :param m: array of halo masses [M_sun]
        :param filter: the window function; TOPHAT or BENSON2026
        :return: sigma(m) at z = 0
        """
        radius = self.lagrangian_radius(m)
        delta2 = self._delta2(self._pk)
        sigma2 = np.empty(len(m))
        if filter == 'BENSON2026':
            c_w, beta = self._benson_window_parameters()
        # evaluate in chunks of masses to bound the memory footprint of the (n_mass, n_k) window array
        n_chunk = 50
        for i in range(0, len(m), n_chunk):
            x = np.outer(radius[i:i + n_chunk], self._k)
            if filter == 'TOPHAT':
                w = self._tophat(x)
            else:
                w = self._benson_window(x / c_w, beta, self._kwargs_benson_window['x_min'])
            sigma2[i:i + n_chunk] = np.trapezoid(delta2 * w ** 2, self._lnk, axis=1)
        return np.sqrt(sigma2)

    def _delta2(self, pk):
        """
        :param pk: power spectrum on the internal grid [Mpc^3]
        :return: the dimensionless power spectrum k^3 P(k) / (2 pi^2)
        """
        return self._k ** 3 * pk / (2 * np.pi ** 2)

    def _benson_window_parameters(self):
        """
        c_W(k) and beta(k) of the Benson et al. (2026) window (their Equation 23), from the logarithmic slope of the
        power spectrum smoothed with a Gaussian kernel in ln k (their Equation 24)
        :return: c_W and beta on the internal wavenumber grid
        """
        p = self._kwargs_benson_window
        n = self.smoothed_logarithmic_slope(p['kwidth'])
        c_w = p['cW0'] * p['cW1'] ** (n - p['n0'])
        beta = p['beta0'] * p['beta1'] ** (n - p['n0'])
        return c_w, beta

    def smoothed_logarithmic_slope(self, kwidth):
        """
        The logarithmic slope n(k) = dln(P_smooth)/dln(k) of the power spectrum convolved with a Gaussian kernel of
        width kwidth in ln k; the derivative of the kernel is computed analytically
        :param kwidth: width of the Gaussian kernel in ln k
        :return: n(k) on the internal wavenumber grid
        """
        dlnk = self._lnk[1] - self._lnk[0]
        # truncate the kernel at 10 standard deviations; np.convolve in 'same' mode needs it no longer than the grid
        n_half = min(int(np.ceil(10 * kwidth / dlnk)), (len(self._lnk) - 1) // 2)
        d = np.arange(-n_half, n_half + 1) * dlnk
        kernel = np.exp(-0.5 * (d / kwidth) ** 2)
        # np.convolve(a, v, 'same')[i] = sum_j a[j] v[i - j + n_half], so v must be indexed by d = ln k_i - ln k_j
        p_smooth = np.convolve(self._pk, kernel, mode='same')
        dp_smooth = np.convolve(self._pk, kernel * (-d / kwidth ** 2), mode='same')
        return dp_smooth / p_smooth

    @staticmethod
    def _tophat(x):
        """
        The real-space top-hat window function
        :param x: k R
        :return: W(x)
        """
        x = np.asarray(x, dtype=float)
        w = np.empty_like(x)
        small = x < 1e-2
        xs = x[small]
        # series expansion where the closed form loses precision to cancellation
        w[small] = 1 - xs ** 2 / 10 + xs ** 4 / 280
        xl = x[~small]
        w[~small] = 3 * (np.sin(xl) - xl * np.cos(xl)) / xl ** 3
        return w

    @staticmethod
    def _benson_window(x, beta, x_min):
        """
        The Benson et al. (2026) window function, their Equation 22
        :param x: k R / c_W(k)
        :param beta: beta(k)
        :param x_min: the scale below which the window equals one
        :return: W
        """
        y = np.maximum(x - x_min, 0.0)
        return 1.0 / (1.0 + y ** beta)

    @staticmethod
    def _interpolate_power_spectrum(k, pk, k_eval):
        """
        Interpolates the power spectrum in log-log space, and extrapolates it beyond the table with power laws that
        match the logarithmic slope between the two points at each end of the table
        :param k: tabulated wavenumbers
        :param pk: tabulated power spectrum
        :param k_eval: wavenumbers at which to evaluate the power spectrum
        :return: P(k_eval)
        """
        lnk, lnp = np.log(k), np.log(pk)
        lnk_eval = np.log(k_eval)
        lnp_eval = np.interp(lnk_eval, lnk, lnp)
        slope_low = (lnp[1] - lnp[0]) / (lnk[1] - lnk[0])
        slope_high = (lnp[-1] - lnp[-2]) / (lnk[-1] - lnk[-2])
        low = lnk_eval < lnk[0]
        high = lnk_eval > lnk[-1]
        lnp_eval[low] = lnp[0] + slope_low * (lnk_eval[low] - lnk[0])
        lnp_eval[high] = lnp[-1] + slope_high * (lnk_eval[high] - lnk[-1])
        return np.exp(lnp_eval)
