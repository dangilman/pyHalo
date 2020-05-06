import numpy as np

class Normal(object):

    """
    set up for black holes right now
    """

    def __init__(self, mean, sigma):

        """
        Mass function of the form:

        dN/dm = ( norm/sqrt(2*pi*mean^2) ) * exp(-(m - mean)^2/sigma^2)

        :param mean:
        :param sigma:
        """

        # log_mass_mean is base 10
        self.mean = mean
        self.sigma = sigma
