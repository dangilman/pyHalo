from pyHalo.single_realization import RealizationFast


class TestSingleRealization(object):

    def setup(self):

        masses = [10**8, 10**8]
        x = [0.3, 0.0]
        y = [-2, 0.]
        r2d = [0.3, 0.5]
        r3d = [20, 20]
        mdefs = ['TNFW']*2
        z = [0.5, 1.5]
        subhalo_flag = [False, False]
        self.zlens, self.zsource = 0.5, 1.5
        cone_opening_angle = 6
        self.realization = RealizationFast(masses, x, y, r2d, r3d, mdefs, z, subhalo_flag, self.zlens, self.zsource,
                 cone_opening_angle, log_mlow=6, log_mhigh=10, mass_sheet_correction=False)


#t.test_mass_at_z()
#
# if __name__ == '__main__':
#     pytest.main()



