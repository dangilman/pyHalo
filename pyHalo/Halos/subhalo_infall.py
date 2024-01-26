import numpy as np
from scipy.interpolate import interp1d
from scipy.special import erfc


def load_subhalo_infall_model(z_lens=None, model='GALACTICUS2020', custom_infall_model=None):
    """
    Sets the model used to assign subhalos an infall redshift
    :param z_lens: the main deflector redshift
    :param model: a string that defines a particular model
    :param custom_infall_model: a custom function or class that takes as input subhalo mass and returns an
    infall redshift
    :return: a function or class with a call method that takes as input subhalo mass at infall and returns infall z
    """
    if model == 'GALACTICUS2020':
        return GalacticusInfall2020(z_lens)
    elif model == 'CUSTOM':
        return custom_infall_model
    else:
        raise Exception('infall model '+str(model)+' not implemented!')

class GalacticusInfall2020(object):
    """
    This class implements the infall model used in Gilman 2020 that is based on the version of Galacticus
    used in that work
    """
    def __init__(self, z_lens):

        self.z_lens = z_lens
        self._computed_zacc_pdf = False

    def __call__(self, msub):

        mlist, dzvals, cdfs = self._subhalo_accretion_pdfs
        idx = self._mass_index(msub, mlist)
        z_accreted = self.z_lens + self._sample_cdf_single(cdfs[idx])
        return z_accreted

    @property
    def _subhalo_accretion_pdfs(self):

        if self._computed_zacc_pdf is False:
            self._computed_zacc_pdf = True
            self._mlist, self._dzvals, self._cdfs = self._Msub_cdfs(self.z_lens)

        return self._mlist, self._dzvals, self._cdfs

    def _cdf_numerical(self, m, z_lens, delta_z_values):

        c_d_f = []
        prob = 0
        for zi in delta_z_values:
            prob += self._P_fit_diff_M_sub(z_lens + zi, z_lens, m)
            c_d_f.append(prob)
        return np.array(c_d_f) / c_d_f[-1]

    def _Msub_cdfs(self, z_lens):

        M_sub_exp = np.arange(6.0, 10.2, 0.2)
        M_sub_list = 10 ** M_sub_exp
        delta_z = np.linspace(0., 6, 8000)
        funcs = []

        for mi in M_sub_list:
            # cdfi = P_fit_diff_M_sub_cumulative(z_lens+delta_z, z_lens, mi)
            cdfi = self._cdf_numerical(mi, z_lens, delta_z)

            funcs.append(interp1d(cdfi, delta_z))

        return M_sub_list, delta_z, funcs

    @staticmethod
    def z_decay_mass_dependence(M_sub):
        # Mass dependence of z_decay.
        a = 3.21509397
        b = 1.04659814e-03
        return a - b * np.log(M_sub / 1.0e6) ** 3

    @staticmethod
    def z_decay_exp_mass_dependence(M_sub):
        # Mass dependence of z_decay_exp.
        a = 0.30335749
        b = 3.2777e-4
        return a - b * np.log(M_sub / 1.0e6) ** 3

    def _P_fit_diff_M_sub(self, z, z_lens, M_sub):
        # Given the redhsift of the lens, z_lens, and the subhalo mass, M_sub, return the
        # posibility that the subhlao has an accretion redhisft of z.
        z_decay = self.z_decay_mass_dependence(M_sub)
        z_decay_exp = self.z_decay_exp_mass_dependence(M_sub)

        normalization = 2.0 / np.sqrt(2.0 * np.pi) / z_decay \
                        / np.exp(0.5 * z_decay ** 2 * z_decay_exp ** 2) \
                        / erfc(z_decay * z_decay_exp / np.sqrt(2.0))
        return normalization * np.exp(-0.5 * ((z - z_lens) / z_decay) ** 2) \
               * np.exp(-z_decay_exp * (z - z_lens))

    @staticmethod
    def _sample_cdf_single(cdf_interp):
        u = np.random.uniform(0, 1)
        try:
            output = float(cdf_interp(u))
            if np.isnan(output):
                output = 0
        except:
            output = 0
        return output

    @staticmethod
    def _mass_index(subhalo_mass, mass_array):
        idx = np.argmin(np.absolute(subhalo_mass - mass_array))
        return idx
