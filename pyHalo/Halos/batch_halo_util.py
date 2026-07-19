"""
batch_halo_util.py

Vectorized, batched evaluation of per-halo quantities in pyHalo. The design
philosophy is "warm the caches": every Halo class in pyHalo already caches
computed quantities in private attributes (_c, _nfw_params, _z_infall,
_time_since_infall, _profile_args, ...). These routines compute the same
quantities for a whole list of halos with vectorized numpy operations and then
store the results in those same attributes. No Halo class API changes; halos
that are not precomputed continue to work exactly as before, and downstream
code (lenstronomy_params, bound_mass, etc.) is unchanged.

Intended usage (e.g. inside pyHalo.Halos.util or a new module pyHalo/Halos/batch_util.py):

    from pyHalo.Halos.batch_util import precompute_tnfw_subhalos, precompute_nfw_params

    # after creating a realization:
    precompute_tnfw_subhalos(realization.subhalos, truncation_class)
    precompute_nfw_params(realization.field_halos)

RNG note: routines that involve random draws (infall redshifts, concentration
scatter, galacticus mass loss) draw the same distributions with the same
scipy/numpy generators, but as array draws rather than one draw per halo. The
realizations are statistically identical to the per-halo code path, but are
not reproducible draw-for-draw against it for a fixed numpy seed.
"""
import numpy as np
from scipy.stats import johnsonsu, truncnorm


# ----------------------------------------------------------------------------
# 1) NFW parameters (rhos, rs, r200): vectorized version of
#    LensCosmo.NFW_params_physical / lenstronomy NFWParam
# ----------------------------------------------------------------------------

def nfw_params_physical_vectorized(lens_cosmo, m, c, z, pseudo_nfw=False):
    """
    Vectorized replica of LensCosmo.NFW_params_physical. Bitwise-identical to
    the scalar version because astropy's efunc accepts arrays.

    :param lens_cosmo: an instance of LensCosmo
    :param m: array of masses [M_sun]
    :param c: array of concentrations
    :param z: array of redshifts
    :param pseudo_nfw: bool or boolean array; NFW or pseudo-NFW normalization
    :return: rhos [M_sun/kpc^3], rs [kpc], r200 [kpc] arrays
    """
    m = np.atleast_1d(np.asarray(m, dtype=float))
    c = np.atleast_1d(np.asarray(c, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    h = lens_cosmo.cosmo.h
    nfw_param = lens_cosmo._nfw_param
    # critical density at z in h^2 M_sun / Mpc^3 (physical); efunc broadcasts
    rhoc_z = nfw_param.rhoc * lens_cosmo.cosmo.astropy.efunc(z) ** 2
    # r200 in physical Mpc (r200_M takes M in M_sun/h, returns Mpc/h)
    r200_mpc = (3 * (m * h) / (4 * np.pi * rhoc_z * 200)) ** (1. / 3.) / h
    # density normalization in physical M_sun/Mpc^3
    pseudo_nfw = np.broadcast_to(np.asarray(pseudo_nfw, dtype=bool), m.shape)
    rho0_nfw = 200. / 3 * rhoc_z * c ** 3 / (np.log(1 + c) - c / (1 + c))
    rho0_pseudo = 2 * c ** 3 * rhoc_z * 200 / (3 * np.log(1 + c ** 2))
    rho0_mpc = np.where(pseudo_nfw, rho0_pseudo, rho0_nfw) * h ** 2
    rs_mpc = r200_mpc / c
    # convert to kpc units exactly as NFW_params_physical does
    return rho0_mpc * 1000 ** -3, rs_mpc * 1000, r200_mpc * 1000


def precompute_nfw_params(halos):
    """
    Batch-computes (rhos, rs, r200) for a list of halos and stores the result
    in each halo's _nfw_params cache. Requires concentrations; computes them
    in batch first (see precompute_concentrations).

    :param halos: list of Halo instances that define nfw_params (NFW/TNFW/TNFWC etc.)
    """
    halos = [h for h in halos if h._pseudo_nfw is not None]
    if len(halos) == 0:
        return
    precompute_concentrations(halos)
    lens_cosmo = halos[0].lens_cosmo
    m = np.array([h.mass for h in halos])
    c = np.array([h.c for h in halos])
    z_eval = np.array([h.z_eval for h in halos])
    pseudo = np.array([h._pseudo_nfw for h in halos])
    rhos, rs, r200 = nfw_params_physical_vectorized(lens_cosmo, m, c, z_eval, pseudo)
    for i, halo in enumerate(halos):
        halo._nfw_params = [rhos[i], rs[i], r200[i]]


# ----------------------------------------------------------------------------
# 2) Concentrations: batch calls into colossus grouped by redshift
# ----------------------------------------------------------------------------

# concentration models whose _evaluate_concentration accepts an array of
# masses at fixed z (verified against pyHalo/Halos/concentration.py):
# DiemerJoyce and Ludlow pass M straight to colossus, PeakHeight uses
# colossus peaks.peakHeight, all array-safe. BinnedHaloMass is NOT (requires
# scalar M), and custom user classes are unknown, so anything not listed
# falls back to the per-halo loop.
_ARRAY_SAFE_CONCENTRATION_MODELS = ('DIEMERJOYCE19', 'LUDLOW2016', 'PEAK_HEIGHT_POWERLAW')


def batch_nfw_concentration(c_class, m, z, force_no_scatter=False):
    """
    Evaluates a pyHalo concentration class for an array of masses at fixed
    redshift with a single call into colossus where the model allows it,
    bypassing the per-mass loop in _ConcentrationCDM.nfw_concentration.
    Replicates the scatter and universal-minimum logic of the base class
    exactly (scatter as one vectorized lognormal draw). WDM turnover models
    (WDM_POLYNOMIAL, WDM_HYPERBOLIC, LUDLOW_WDM) are handled by evaluating
    their CDM class in batch and applying the (vectorized) suppression.
    Unknown models fall back to the per-halo path.

    :param c_class: an instantiated pyHalo concentration class
    :param m: array of halo masses [M_sun]
    :param z: redshift (scalar)
    :param force_no_scatter: bool; return the median relation
    :return: array of concentrations
    """
    m = np.atleast_1d(np.asarray(m, dtype=float))
    name = getattr(c_class, 'name', None)
    if name in _ARRAY_SAFE_CONCENTRATION_MODELS:
        c = np.atleast_1d(np.asarray(c_class._evaluate_concentration(m, z), dtype=float))
        if c_class._scatter and not force_no_scatter:
            sigma = c_class._scatter_dex + c_class._scatter_dex_z_dep * z
            c = np.random.lognormal(np.log(c), sigma)
        c[c < c_class._universal_minimum] = c_class._universal_minimum
        return c
    if hasattr(c_class, '_cdm_concentration') and hasattr(c_class, 'suppression'):
        c_cdm = batch_nfw_concentration(c_class._cdm_concentration, m, z, force_no_scatter)
        c_wdm = c_cdm * c_class.suppression(m, z)
        c_wdm = np.atleast_1d(c_wdm)
        c_wdm[c_wdm < c_class._universal_minimum] = c_class._universal_minimum
        return c_wdm
    if name == 'CONSTANT':
        return np.full(len(m), c_class.nfw_concentration())
    return np.array([c_class.nfw_concentration(float(mi), z) for mi in m])


def precompute_concentrations(halos):
    """
    Batch-computes halo concentrations and stores them in the _c cache.

    Halos are grouped by (concentration class, z_eval) and evaluated through
    batch_nfw_concentration, so each group costs one call into the model.
    Field halos live on discrete lens planes so they batch well; subhalos are
    evaluated per unique infall redshift, which is fast when the concentration
    class carries the internal c(m, z) interpolation table
    (use_interpolation_table=True, the default in the speed-patched
    concentration.py).

    Scatter, when the concentration class carries it, is applied as a single
    vectorized lognormal draw (same distribution as the per-halo draws).
    """
    todo = [h for h in halos if not hasattr(h, '_c')]
    if len(todo) == 0:
        return
    groups = {}
    for h in todo:
        key = (id(h._concentration_class), float(h.z_eval))
        groups.setdefault(key, []).append(h)
    for (_, z), group in groups.items():
        c_class = group[0]._concentration_class
        m = np.array([h.mass for h in group])
        if len(group) == 1:
            group[0]._c = c_class.nfw_concentration(float(m[0]), z)
            continue
        c = batch_nfw_concentration(c_class, m, z)
        for i, h in enumerate(group):
            h._c = float(c[i])


# ----------------------------------------------------------------------------
# 3) Subhalo infall redshifts and times since infall
# ----------------------------------------------------------------------------

def precompute_infall_times(subhalos, lens_cosmo):
    """
    Batch-computes z_infall and time_since_infall for subhalos, storing them
    in the _z_infall and _time_since_infall caches.

    - Infall redshifts are drawn with a single vectorized truncnorm call for
      the Hybrid/Direct galacticus infall models (same distribution as the
      per-halo draws). For other models, falls back to per-halo sampling.
    - time_since_infall uses one vectorized astropy age(z) call for all halos
      instead of two scalar astropy.age calls per halo; the numbers are
      identical to the per-halo path.
    """
    subhalos = [h for h in subhalos if h.is_subhalo]
    if len(subhalos) == 0:
        return
    need_z = [h for h in subhalos if not hasattr(h, '_z_infall')]
    if len(need_z) > 0:
        model = getattr(lens_cosmo, '_z_infall_model', None)
        m = np.array([h.mass for h in need_z])
        z_inf = _sample_infall_redshifts(model, m, lens_cosmo)
        for i, h in enumerate(need_z):
            h._z_infall = float(z_inf[i])
    # vectorized time since infall, matching whichever implementation of
    # Halo.time_since_infall is installed: the speed-patched halo_base uses the
    # interpolated age table (Cosmology.halo_age), unpatched pyHalo calls
    # astropy.age directly
    need_t = [h for h in subhalos if not hasattr(h, '_time_since_infall')]
    if len(need_t) > 0:
        import inspect
        from pyHalo.Halos.halo_base import Halo as _Halo
        _patched = 'halo_age' in inspect.getsource(_Halo.time_since_infall.fget)
        z = np.array([h.z for h in need_t])
        z_infall = np.array([h._z_infall for h in need_t])
        if _patched:
            age_interp = lens_cosmo.cosmo._halo_age_interp
            dt = np.where(z > z_infall, 0.0, age_interp(z) - age_interp(z_infall))
        else:
            astropy = lens_cosmo.cosmo.astropy
            dt = astropy.age(z).value - astropy.age(z_infall).value
        for i, h in enumerate(need_t):
            assert dt[i] >= 0
            h._time_since_infall = float(dt[i])


def _sample_infall_redshifts(model, m, lens_cosmo):
    """
    Vectorized infall-redshift sampling for the galacticus-calibrated
    Hybrid/Direct models (truncnorm accepts array loc/scale). Any other model,
    or any inconsistency in the vectorized draw (e.g. an infall model whose
    internals are not array-safe), falls back to the public per-halo API,
    which is always correct.
    """
    m = np.atleast_1d(np.asarray(m, dtype=float))
    name = getattr(model, 'name', None)
    if name in ('hybrid', 'direct') and hasattr(model, 'z_inf_to_z_host_mean'):
        try:
            mass_ratio = np.clip(m / model._m_host, 10 ** -5.0, 10 ** -0.5)
            mu = np.atleast_1d(np.asarray(model.z_inf_to_z_host_mean(mass_ratio), dtype=float))
            sig = np.atleast_1d(np.asarray(model.z_inf_to_z_host_std(mass_ratio), dtype=float))
            if len(mu) not in (1, len(m)) or len(sig) not in (1, len(m)):
                raise ValueError('infall model returned unexpected shapes')
            mu = np.broadcast_to(mu, m.shape)
            sig = np.broadcast_to(sig, m.shape)
            bounds = [0.0, 15.0]
            z = truncnorm.rvs((bounds[0] - mu) / sig, (bounds[1] - mu) / sig,
                              loc=mu, scale=sig, size=len(m))
            z_inf = np.atleast_1d(model._z_lens + z)
            if z_inf.shape == m.shape and np.all(np.isfinite(z_inf)):
                return z_inf
        except Exception:
            pass
    return np.array([lens_cosmo.z_accreted_from_zlens(float(mi)) for mi in m])


# ----------------------------------------------------------------------------
# 4) Galacticus tidal truncation: batch bound masses, r_te and f_t
# ----------------------------------------------------------------------------

def galacticus_mass_loss_vectorized(mass_loss_interp, log10c, t_or_dz, chost):
    """
    Vectorized version of InterpGalacticus.__call__ /
    InterpGalacticusZinfall.__call__: evaluates the JohnsonSU (a, b)
    interpolators at all points in one call and draws all samples with a
    single array johnsonsu.rvs call.

    :param mass_loss_interp: the _mass_loss_interp attribute of a
        TruncationGalacticus(-Zinfall) class
    :param log10c: array of log10 infall concentrations
    :param t_or_dz: array of times since infall [Gyr] (InterpGalacticus) or
        elapsed redshift since infall (InterpGalacticusZinfall)
    :param chost: host concentration (scalar)
    :return: array of log10(m_bound / m_infall), clipped at 0
    """
    cls = type(mass_loss_interp).__name__
    if cls == 'InterpGalacticus':
        log10c = np.clip(log10c, np.log10(2.), np.log10(384.))
        t_or_dz = np.clip(t_or_dz, 0.0, 8.1)
        chost = min(max(chost, 4.0), 9.0)
    elif cls == 'InterpGalacticusZinfall':
        log10c = np.clip(log10c, np.log10(2.), np.log10(384.))
        t_or_dz = np.clip(t_or_dz, 0.001, 9.5)
        chost = min(max(chost, 4.0), 8.0)
    else:
        # generic clipping to the interpolation domain
        pass
    n = len(log10c)
    pts = np.column_stack((t_or_dz, log10c, np.full(n, chost)))
    a = mass_loss_interp._a_interp(pts)
    b = mass_loss_interp._b_interp(pts)
    out = johnsonsu.rvs(a, b)
    return np.minimum(out, 0.0)


def compute_r_te_and_f_t_vectorized(m_bound, m_infall, r200_infall, c_infall):
    """
    Vectorized version of
    pyHalo.Halos.galacticus_truncation.transfer_function_density_profile.compute_r_te_and_f_t
    for the standard NFW case (alpha, beta, gamma, delta) = (1, 3, 1, 2).

    All the underlying formulas (Du et al. 2024 fitting functions, the
    enclosed-mass expression, and the mass table used to invert M(tau)) are
    already numpy array-safe; the expensive parts, which the per-halo code
    repeats on every call, are done exactly once here:
      - the reference-model conversion factors (constants),
      - the mass table and its log-log cubic interpolator.

    :return: r_te [kpc], f_t arrays
    """
    from pyHalo.Halos.galacticus_truncation import transfer_function_density_profile as tf
    m_bound = np.atleast_1d(np.asarray(m_bound, dtype=float))
    m_infall = np.atleast_1d(np.asarray(m_infall, dtype=float))
    r200_infall = np.atleast_1d(np.asarray(r200_infall, dtype=float))
    c_infall = np.atleast_1d(np.asarray(c_infall, dtype=float))

    rs = r200_infall / c_infall
    rmax_dimensionless = tf._R_max_generalized_NFW_dimensionless(1.0, 3.0, 1.0)
    mu = lambda x: tf._M_enclosed_generalized_NFW_dimensionless(x, 1.0, 3.0, 1.0)
    m0 = m_infall / mu(c_infall)
    m_mx = m0 * mu(rmax_dimensionless)
    y = m_bound / m_mx
    y_scale, _ = _reference_model_factors()
    f_t = tf._f_t_Du_2024(y * y_scale, 1.0, 3.0, 1.0, 2.0)
    m_bound_dimensionless = m_bound / m0
    r_te = rs * _truncation_radius_cached(m_bound_dimensionless, f_t)
    return r_te, f_t


_REF_FACTORS = None

def _reference_model_factors():
    global _REF_FACTORS
    if _REF_FACTORS is None:
        from pyHalo.Halos.galacticus_truncation import transfer_function_density_profile as tf
        _REF_FACTORS = tf.Convert_to_reference_model(1.0, 3.0, 1.0)
    return _REF_FACTORS


_TRUNC_TABLE = {'interp': None, 'logM_min': None, 'logM_max': None}

def _truncation_radius_cached(M, f_t):
    """
    Same computation as transfer_function_density_profile._Truncation_Radius
    for (1, 3, 1, 2), but the mass table AND the cubic log-log interpolator are
    built once and reused; the per-halo implementation rebuilds the cubic
    spline on every call, which dominates its runtime.
    """
    from scipy.interpolate import interp1d
    from pyHalo.Halos.galacticus_truncation import transfer_function_density_profile as tf
    M_unnormalized = np.asarray(M, dtype=float) / np.asarray(f_t, dtype=float)
    logM = np.log(M_unnormalized)
    tab = _TRUNC_TABLE
    if (tab['interp'] is None or np.min(logM) < tab['logM_min']
            or np.max(logM) > tab['logM_max']):
        x_min, x_max = 1.0e-6, 1.0e+4
        M_min = tf._M_total_generalized_NFW_Truncated_dimensionless(x_min, 1.0, 3.0, 1.0, 2.0)
        M_max = tf._M_total_generalized_NFW_Truncated_dimensionless(x_max, 1.0, 3.0, 1.0, 2.0)
        while np.min(M_unnormalized) < M_min:
            x_min /= 2.0
            M_min = tf._M_total_generalized_NFW_Truncated_dimensionless(x_min, 1.0, 3.0, 1.0, 2.0)
        while np.max(M_unnormalized) > M_max:
            x_max *= 2.0
            M_max = tf._M_total_generalized_NFW_Truncated_dimensionless(x_max, 1.0, 3.0, 1.0, 2.0)
        n_tab = int(np.log10(x_max / x_min) * 10)
        x_tab = np.geomspace(x_min, x_max, n_tab)
        m_tab = tf._M_total_generalized_NFW_Truncated_dimensionless(x_tab, 1.0, 3.0, 1.0, 2.0)
        tab['interp'] = interp1d(np.log(m_tab), np.log(x_tab), kind='cubic')
        tab['logM_min'], tab['logM_max'] = np.log(m_tab[0]), np.log(m_tab[-1])
    return np.exp(tab['interp'](logM))


def precompute_tnfw_subhalos(subhalos, truncation_class):
    """
    Full batch precomputation for TNFWSubhalo objects using one of the
    TruncationGalacticus classes. After this call, profile_args,
    lenstronomy_params, bound_mass etc. read from warmed caches and involve no
    further expensive computation.

    Order of operations mirrors the per-halo chain exactly:
      z_infall -> time_since_infall -> c(z_infall) -> mass loss -> NFW params
      -> (r_te, f_t) -> rescale_normalization(f_t)

    :param subhalos: list of TNFWSubhalo instances
    :param truncation_class: the shared truncation class instance
        (TruncationGalacticus or TruncationGalacticusZinfall)
    """
    subhalos = [h for h in subhalos if h.is_subhalo and not hasattr(h, '_profile_args')]
    if len(subhalos) == 0:
        return
    lens_cosmo = subhalos[0].lens_cosmo
    tname = truncation_class.name

    precompute_infall_times(subhalos, lens_cosmo)
    precompute_concentrations(subhalos)

    m = np.array([h.mass for h in subhalos])
    c = np.array([h.c for h in subhalos])
    log10c = np.log10(c)

    if tname == 'TruncationGalacticus':
        t_or_dz = np.array([h.time_since_infall for h in subhalos])
    elif tname == 'TruncationGalacticusZinfall':
        t_or_dz = np.array([h.z_infall for h in subhalos]) - lens_cosmo.z_lens
    else:
        raise Exception('precompute_tnfw_subhalos supports the TruncationGalacticus '
                        'and TruncationGalacticusZinfall models, received ' + str(tname))

    # bound masses for halos that do not already have one assigned
    m_bound = np.empty(len(subhalos))
    has_mb = np.array([hasattr(h, '_mbound_galacticus_definition') for h in subhalos])
    if np.any(~has_mb):
        log10_mloss = galacticus_mass_loss_vectorized(truncation_class._mass_loss_interp,
                                                      log10c[~has_mb],
                                                      t_or_dz[~has_mb],
                                                      truncation_class._chost)
        m_bound[~has_mb] = m[~has_mb] * 10 ** log10_mloss
    for i, h in enumerate(subhalos):
        if has_mb[i]:
            m_bound[i] = h._mbound_galacticus_definition
        else:
            h._mbound_galacticus_definition = m_bound[i]

    # NFW params at infall (pseudo_nfw=False for TNFW)
    z_eval = np.array([h.z_eval for h in subhalos])
    rhos, rs, r200 = nfw_params_physical_vectorized(lens_cosmo, m, c, z_eval, False)

    r_te, f_t = compute_r_te_and_f_t_vectorized(m_bound, m, r200, c)

    for i, h in enumerate(subhalos):
        h.rescale_normalization(f_t[i])
        h._nfw_params = [rhos[i], rs[i], r200[i]]
        h._profile_args = (c[i], r_te[i])


def precompute_tnfw_field_halos(field_halos, truncation_class):
    """
    Batch precomputation for TNFWFieldHalo objects with a truncation model
    whose per-halo work reduces to NFW parameters (Multiple_RS, TruncationRN).
    Other truncation models still benefit from the batched concentrations and
    NFW parameters computed here, after which truncation_radius_halo is
    evaluated per halo as before.
    """
    field_halos = [h for h in field_halos if not h.is_subhalo
                   and not hasattr(h, '_profile_args')]
    if len(field_halos) == 0:
        return
    precompute_nfw_params(field_halos)
    name = getattr(truncation_class, 'name', None)
    if name == 'Multiple_RS':
        for h in field_halos:
            h._profile_args = (h.c, truncation_class.tau * h._nfw_params[1])
    elif name == 'TruncationRN':
        lens_cosmo = field_halos[0].lens_cosmo
        m = np.array([h.mass for h in field_halos])
        z = np.array([h.z_eval for h in field_halos])
        h_cosmo = lens_cosmo.cosmo.h
        rhoc_z = lens_cosmo._nfw_param.rhoc * lens_cosmo.cosmo.astropy.efunc(z) ** 2
        # replicate TruncationRN.truncation_radius, which evaluates
        # rN_M(halo_mass * h, z, N) = (3 M / (4 pi rhoc_z N))^(1/3) / h with M = m * h
        rn = (3 * m * h_cosmo / (4 * np.pi * rhoc_z * truncation_class._N)) ** (1. / 3.) / h_cosmo
        for i, h in enumerate(field_halos):
            h._profile_args = (h.c, rn[i] * 1000)
    else:
        for h in field_halos:
            h._profile_args = (h.c, truncation_class.truncation_radius_halo(h))


# ----------------------------------------------------------------------------
# 5) TNFW bound masses (vectorized tnfw_mass_fraction)
# ----------------------------------------------------------------------------

def precompute_tnfw_bound_masses(subhalos):
    """
    Batch-computes the .bound_mass attribute of TNFWSubhalo objects (mass
    within the infall virial radius including truncation) and stores it via
    set_bound_mass. Requires _profile_args/_nfw_params (run
    precompute_tnfw_subhalos first). Uses a single vectorized call to
    lenstronomy's TNFW.mass_3d, identical in value to the per-halo path.
    """
    from lenstronomy.LensModel.Profiles.tnfw import TNFW
    subhalos = [h for h in subhalos if h.is_subhalo and not hasattr(h, '_bound_mass')]
    if len(subhalos) == 0:
        return
    c = np.array([h.c for h in subhalos])
    rt = np.array([h.profile_args[1] for h in subhalos])
    rs = np.array([h.nfw_params[1] for h in subhalos])
    m = np.array([h.mass for h in subhalos])
    rescale = np.array([h._rescale_norm for h in subhalos])
    tau = rt / rs
    prof = TNFW()
    m_frac = prof.mass_3d(c, 1.0, 1 / (4 * np.pi), tau) / (np.log(1 + c) - c / (1 + c))
    m_bound = m_frac * m * rescale
    for i, h in enumerate(subhalos):
        h.set_bound_mass(float(m_bound[i]))


# ----------------------------------------------------------------------------
# 6) SIDM evolving profiles (TNFWCHaloEvolving): batch the mass-conservation
#    integral and the profile evolution
# ----------------------------------------------------------------------------

def _evolve_profile_vectorized(t, rs_0):
    """
    Vectorized version of NFW_core_trunc.evolve_profile / rs_evolution /
    rc_evolution, including the tr > 1 extrapolation branches.
    """
    from pyHalo.Halos.HaloModels import NFW_core_trunc as nct
    t = np.minimum(1.6, np.asarray(t, dtype=float))
    below = t <= 1.0
    rs_evo = np.where(below,
                      nct._rs_evolution(t),
                      nct._rs_evolution(0.5) * np.exp(-0.47 * (t - 0.5) - 2.4 * (t - 0.5) ** 3))
    rc_evo = np.where(below,
                      nct._rc_evolution(t),
                      np.maximum(nct._rc_evolution(0.5) - 0.28 * (t - 0.5), 0.0001))
    rs = rs_0 * rs_evo
    rc = np.maximum(rs_0 * rc_evo, 1e-7 * rs)
    return rs, rc


def precompute_sidm_evolving_profiles(halos, n_r=250):
    """
    Batch precomputation of TNFWCHaloEvolving.profile_args. The expensive
    per-halo pieces are (i) two interpolated halo-age lookups, (ii) the NFW
    parameter evaluation, and (iii) a 1000-point trapezoidal integral of the
    density profile for the mass-conservation normalization. All are done
    here as single array operations; the integral becomes one (N, n_r) numpy
    computation using lenstronomy's TNFWC density_lens, which broadcasts.

    :param halos: list of TNFWCHaloEvolving instances
    :param n_r: number of radial points in the mass integral; must match the
        value used in TNFWCHaloEvolving.profile_args for exact agreement with
        the per-halo path (250 after the standalone speed patches, 1000 in
        unpatched pyHalo)
    """
    from lenstronomy.LensModel.Profiles.nfw_core_truncated import TNFWC
    halos = [h for h in halos if h.mdef == 'TNFWC' and not hasattr(h, '_profile_args')]
    if len(halos) == 0:
        return
    lens_cosmo = halos[0]._lens_cosmo
    cosmo = lens_cosmo.cosmo
    precompute_infall_times([h for h in halos if h.is_subhalo], lens_cosmo)
    precompute_concentrations(halos)

    n = len(halos)
    m = np.array([h.mass for h in halos])
    c = np.array([h.c for h in halos])
    z = np.array([h.z for h in halos])
    z_eval = np.array([h.z_eval for h in halos])
    rt_kpc = np.array([h._args['rt_kpc'] for h in halos])
    m_target = np.array([h._args['mass_conservation'] for h in halos])
    t_c = np.array([h._args['sidm_timescale'] for h in halos])

    # halo effective age (vectorized over the same interpolated age table)
    age_interp = cosmo._halo_age_interp
    is_sub = np.array([h.is_subhalo for h in halos])
    eff_age = np.empty(n)
    if np.any(is_sub):
        lam = np.array([h._args['lambda_t'] for h in halos])[is_sub]
        z_inf = np.minimum(np.array([h.z_infall for h in halos if h.is_subhalo]), 10.0)
        t_form_to_infall = np.where(z_inf > 10.0, 0.0, age_interp(z_inf) - age_interp(10.0))
        zi = z[is_sub]
        t_infall_to_z = np.where(zi > z_inf, 0.0, age_interp(zi) - age_interp(z_inf))
        eff_age[is_sub] = t_form_to_infall + lam * t_infall_to_z
    if np.any(~is_sub):
        zi = z[~is_sub]
        eff_age[~is_sub] = np.where(zi > 10.0, 0.0, age_interp(zi) - age_interp(10.0))

    t_over_tc = eff_age / t_c

    # NFW params with the pseudo-NFW normalization used by TNFWCHaloEvolving
    pseudo = np.array([h._pseudo_nfw for h in halos])
    _, rs_0, _ = nfw_params_physical_vectorized(lens_cosmo, m, c, z_eval, pseudo)
    rs_kpc, rc_kpc = _evolve_profile_vectorized(t_over_tc, rs_0)
    r200_kpc = c * rs_0

    # mass-conservation normalization: one broadcasted (N, n_r) integral
    kpc_per_arcsec = np.atleast_1d(np.asarray(cosmo.kpc_proper_per_asec(z)))
    # sigma_crit: vectorized over lens redshift (astropy accepts array z1)
    astropy = cosmo.astropy
    d_ds = astropy.angular_diameter_distance_z1z2(z, lens_cosmo.z_source).value
    d_d = astropy.angular_diameter_distance(z).value
    d_s = astropy.angular_diameter_distance(lens_cosmo.z_source).value
    sigma_crit_mpc = cosmo.c ** 2 / (4 * np.pi * cosmo.G) * d_s / (d_ds * d_d)
    factor = sigma_crit_mpc * 1e-6 / kpc_per_arcsec

    x = np.array([np.logspace(-4, np.log10(ci), n_r) for ci in c])  # (N, n_r)
    r = x * rs_0[:, None]
    prof = TNFWC()
    rs_angle = rs_kpc / kpc_per_arcsec
    rc_angle = rc_kpc / kpc_per_arcsec
    rt_angle = rt_kpc / kpc_per_arcsec
    # alpha2rho0 branches on scalars in lenstronomy, so convert per halo
    # (cheap scalar arithmetic); the density evaluation broadcasts over (N, n_r)
    rho0 = np.array([prof.alpha2rho0(1.0, rs_angle[i], rc_angle[i], rt_angle[i])
                     for i in range(n)])
    rho_lens = prof.density(r / kpc_per_arcsec[:, None],
                            rs_angle[:, None],
                            rho0[:, None],
                            rc_angle[:, None],
                            rt_angle[:, None])
    rho = factor[:, None] * rho_lens
    mass_3d = np.trapezoid(4 * np.pi * r ** 2 * rho, r, axis=1)
    alpha_Rs = m_target / mass_3d

    for i, h in enumerate(halos):
        h._halo_effective_age = float(eff_age[i])
        h._profile_args = (alpha_Rs[i], rs_kpc[i], rc_kpc[i], rt_kpc[i], r200_kpc[i])


# ----------------------------------------------------------------------------
# 7) Convenience driver
# ----------------------------------------------------------------------------

def precompute_realization(realization, kwargs_halo_model=None):
    """
    Warm the caches of every supported halo type in a realization.
    Unsupported halo types are left untouched and behave exactly as before.

    :param realization: an instance of Realization
    :param kwargs_halo_model: the kwargs_halo_model dictionary used to create
        the realization (used to locate the truncation classes); defaults to
        realization.kwargs_halo_model
    """
    if kwargs_halo_model is None:
        kwargs_halo_model = realization.kwargs_halo_model

    def _truncation_for(is_subhalo):
        if 'truncation_model' in kwargs_halo_model:
            return kwargs_halo_model['truncation_model']
        key = 'truncation_model_subhalos' if is_subhalo else 'truncation_model_field_halos'
        return kwargs_halo_model.get(key, None)

    tnfw_subs = [h for h in realization.halos if h.mdef == 'TNFW' and h.is_subhalo]
    tnfw_field = [h for h in realization.halos if h.mdef == 'TNFW' and not h.is_subhalo]
    tnfwc = [h for h in realization.halos if h.mdef == 'TNFWC'
             and 'sidm_timescale' in getattr(h, '_args', {})]
    nfw = [h for h in realization.halos if h.mdef == 'NFW']

    t_sub = _truncation_for(True)
    if len(tnfw_subs) > 0 and t_sub is not None and \
            getattr(t_sub, 'name', None) in ('TruncationGalacticus', 'TruncationGalacticusZinfall'):
        precompute_tnfw_subhalos(tnfw_subs, t_sub)
    t_field = _truncation_for(False)
    if len(tnfw_field) > 0 and t_field is not None:
        precompute_tnfw_field_halos(tnfw_field, t_field)
    if len(tnfwc) > 0:
        precompute_sidm_evolving_profiles(tnfwc)
    if len(nfw) > 0:
        precompute_nfw_params(nfw)
