import numpy as np
import numpy.testing as npt
from pyHalo.Rendering.MassFunctions.benson import benson_suppression_params
from pyHalo.Rendering.MassFunctions.stucker import stucker_suppression_params
from pyHalo.mass_function_models import preset_mass_function_models
from pyHalo.Cosmology.cosmology import Cosmology
from pyHalo.Cosmology.geometry import Geometry
import pytest


class TestBenson(object):

    def setup_method(self):

        self.cosmo = Cosmology()
        self.geometry = Geometry(self.cosmo, 0.5, 2.0, 6.0, 'DOUBLE_CONE')
        self.log_mc = 7.5
        self.kwargs_model = {'log_mlow': 6.0, 'log_mhigh': 10.0, 'm_pivot': 10 ** 8,
                             'delta_power_law_index': 0.0, 'draw_poisson': False,
                             'LOS_normalization': 1.0, 'log_mc': self.log_mc}

    def _mass_function(self, dlogT_dlogk, z, z_eval_suppression=None):
        """
        Builds the mass function through the preset model and from_redshift, so the test
        exercises the class rather than only the calibration routine
        """
        model, kwargs = preset_mass_function_models(
            'BENSON', {'dlogT_dlogk': dlogT_dlogk, 'z_eval_suppression': z_eval_suppression})
        kwargs.update(self.kwargs_model)
        return model.from_redshift(z, 0.02, self.geometry, kwargs)

    def test_suppression_limits(self):
        """
        The Benson turnover must approach CDM well above the half-mode mass, and must agree
        with the Stucker calibration to within 15% at the half-mode mass itself
        """
        m_hm = 10 ** self.log_mc

        # (1) CDM is recovered for m_hm << m, at every cutoff slope and redshift
        for dlogT_dlogk in [-1.0, -2.0, -4.0]:
            for z in [0.0, 5.0, 10.0]:
                mfunc = self._mass_function(dlogT_dlogk, 0.5, z_eval_suppression=z)
                suppression = mfunc.turnover(np.array([1e6 * m_hm, 1e8 * m_hm]))
                # shallow cutoffs approach CDM slowly, so the limit is tested far above
                # m_hm; the worst case over this grid is 0.9943 at 1e6 m_hm
                npt.assert_array_less(0.99, suppression)
                npt.assert_almost_equal(suppression[-1], 1.0, 3)
                # monotonically approaching CDM from below, never exceeding it
                npt.assert_array_less(suppression[0], suppression[-1])
                npt.assert_array_less(suppression[-1], 1.0 + 1e-12)

        # (2) at m = m_hm the suppression matches Stucker to 15%
        for dlogT_dlogk in [-1.0, -1.5, -2.0, -2.5, -3.0, -3.5]:
            mfunc = self._mass_function(dlogT_dlogk, 0.5, z_eval_suppression=0.0)
            benson = float(mfunc.turnover(np.array([m_hm]))[0])
            a, b, c = stucker_suppression_params(dlogT_dlogk)
            stucker = (1 + a) ** c
            npt.assert_array_less(abs(benson / stucker - 1), 0.15)
            # both describe a mass function suppressed by roughly a factor 2-3 at m_hm
            npt.assert_array_less(0.3, benson)
            npt.assert_array_less(benson, 0.5)

    def test_redshift_evaluation(self):
        """
        z_eval_suppression=None evaluates at the lens plane redshift; a value holds it fixed
        """
        m = np.array([10 ** self.log_mc])
        per_plane = self._mass_function(-2.0, 6.0, z_eval_suppression=None).turnover(m)
        fixed_at_6 = self._mass_function(-2.0, 0.5, z_eval_suppression=6.0).turnover(m)
        fixed_at_0 = self._mass_function(-2.0, 0.5, z_eval_suppression=0.0).turnover(m)
        npt.assert_almost_equal(per_plane[0], fixed_at_6[0], 6)
        # the suppression strengthens with redshift
        npt.assert_array_less(per_plane[0], fixed_at_0[0])

    def test_params_out_of_bounds(self):
        """
        Arguments outside the calibration are clipped, and positive slopes are rejected
        """
        npt.assert_allclose(benson_suppression_params(-0.5, 2.0),
                            benson_suppression_params(-1.0, 2.0))
        npt.assert_allclose(benson_suppression_params(-6.0, 2.0),
                            benson_suppression_params(-4.0, 2.0))
        npt.assert_allclose(benson_suppression_params(-2.0, -1.0),
                            benson_suppression_params(-2.0, 0.0))
        npt.assert_allclose(benson_suppression_params(-2.0, 20.0),
                            benson_suppression_params(-2.0, 10.0))
        npt.assert_raises(Exception, benson_suppression_params, 1.0, 0.0)

if __name__ == '__main__':
   pytest.main()
