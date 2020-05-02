import numpy as np

class Gaussian(object):

    """
    set up for black holes right now
    """

    def __init__(self, log_mass_mean, standard_dev, n_draw):

        # log_mass_mean is base 10
        self.log_mass_mean_base_e = np.log(10**log_mass_mean)

        # standard deviation is in dex
        self.std_base_e = np.log(10**standard_dev)
        self.N_draw = n_draw

    def draw(self):

        log_masses_base_e = np.random.normal(self.log_mass_mean_base_e, self.std_base_e, self.N_draw)

        log_masses = np.log10(np.exp(log_masses_base_e))

        return 10 ** log_masses

