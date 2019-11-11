import numpy as np
from scipy.interpolate import interp2d

rhos_min = 7*10**6
rhos_max = 2*10**8

rs_min = 0.1
rs_max = 5
zeta_bins = np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])*10
logrhos_values = np.linspace(np.log10(rhos_min), np.log10(rhos_max), 6)
logrs_values = np.linspace(np.log10(rs_min), np.log10(rs_max), 6)
rho_bins = 10**logrhos_values
rs_bins = 10**logrs_values
logrhoarray, logrsarray = np.meshgrid(logrhos_values, logrs_values)

interp_tables = []

coords = np.vstack([logrhoarray.ravel(),logrsarray.ravel()]).T

ratio_table = 10**np.array([[[-1.216, -0.737, -0.415, -0.272, -0.114, -0.031,  0.038,  0.13 ,
          0.145,  0.211,  0.237,  0.35 ,  0.328],
        [-1.216, -0.735, -0.417, -0.272, -0.115, -0.033,  0.037,  0.13 ,
          0.144,  0.21 ,  0.25 ,  0.375,  0.327],
        [-1.241, -0.776, -0.424, -0.288, -0.131, -0.09 ,  0.057,  0.095,
          0.098,  0.156,  0.221,  0.271,  0.292],
        [-1.259, -1.019, -0.82 , -0.633, -0.465, -0.38 , -0.295, -0.241,
         -0.275, -0.184, -0.212, -0.091, -0.031],
        [-2.248, -2.065, -1.639, -1.696, -1.528, -1.495, -1.456, -1.362,
         -1.069, -1.204, -1.219, -1.223, -1.12 ],
        [-3.393, -3.285, -3.04 , -2.796, -2.7  , -2.696, -2.705, -2.499,
         -2.65 , -2.586, -2.491, -2.503, -2.427]],

       [[-0.728, -0.439, -0.109,  0.029,  0.128,  0.249,  0.331,  0.365,
          0.416,  0.469,  0.511,  0.549,  0.584],
        [-0.725, -0.439, -0.136,  0.026,  0.153,  0.193,  0.342,  0.361,
          0.433,  0.465,  0.491,  0.53 ,  0.573],
        [-0.75 , -0.461, -0.206,  0.012,  0.079,  0.157,  0.272,  0.328,
          0.39 ,  0.427,  0.485,  0.5  ,  0.512],
        [-1.065, -0.924, -0.68 , -0.542, -0.428, -0.399, -0.32 , -0.218,
         -0.19 , -0.107, -0.007, -0.037, -0.025],
        [-2.21 , -1.964, -1.84 , -1.728, -1.652, -1.604, -1.527, -1.411,
         -1.395, -1.311, -1.311, -1.269, -1.231],
        [-3.675, -3.4  , -3.198, -2.974, -2.921, -2.755, -2.599, -2.633,
         -2.563, -2.476, -2.509, -2.509, -2.445]],

       [[-0.467, -0.16 ,  0.131,  0.269,  0.398,  0.474,  0.583,  0.624,
          0.697,  0.717,  0.735,  0.802,  0.829],
        [-0.469, -0.143,  0.144,  0.266,  0.394,  0.49 ,  0.578,  0.629,
          0.684,  0.707,  0.772,  0.772,  0.845],
        [-0.5  , -0.243,  0.075,  0.172,  0.327,  0.386,  0.435,  0.509,
          0.59 ,  0.603,  0.679,  0.684,  0.719],
        [-1.128, -0.888, -0.602, -0.542, -0.435, -0.372, -0.264, -0.159,
         -0.158, -0.069, -0.042, -0.068, -0.027],
        [-2.164, -2.113, -1.8  , -1.69 , -1.569, -1.598, -1.553, -1.522,
         -1.453, -1.404, -1.35 , -1.266, -1.304],
        [-3.673, -3.518, -3.193, -3.044, -2.893, -2.808, -2.488, -2.565,
         -2.652, -2.706, -2.59 , -2.465, -2.416]],

       [[-0.133,  0.113,  0.394,  0.551,  0.65 ,  0.731,  0.817,  0.874,
          0.961,  0.995,  0.994,  1.028,  1.075],
        [-0.123,  0.108,  0.363,  0.54 ,  0.654,  0.725,  0.735,  0.86 ,
          0.91 ,  0.969,  0.998,  1.036,  1.067],
        [-0.273, -0.027,  0.246,  0.346,  0.471,  0.554,  0.601,  0.689,
          0.727,  0.799,  0.858,  0.907,  0.902],
        [-1.127, -0.957, -0.643, -0.457, -0.35 , -0.278, -0.25 , -0.229,
         -0.157, -0.137, -0.131, -0.099, -0.031],
        [-2.327, -2.021, -1.9  , -1.77 , -1.674, -1.553, -1.46 , -1.424,
         -1.307, -1.215, -1.29 , -1.331, -1.389],
        [-3.795, -3.472, -3.1  , -3.077, -3.005, -2.931, -2.924, -2.763,
         -2.774, -2.645, -2.608, -2.549, -2.574]],

       [[ 0.102,  0.396,  0.624,  0.808,  0.92 ,  0.976,  1.017,  1.142,
          1.165,  1.199,  1.28 ,  1.283,  1.341],
        [ 0.119,  0.347,  0.633,  0.757,  0.903,  0.961,  1.01 ,  1.094,
          1.165,  1.196,  1.237,  1.262,  1.314],
        [-0.16 ,  0.098,  0.377,  0.516,  0.605,  0.675,  0.787,  0.873,
          0.883,  0.913,  0.927,  0.993,  1.046],
        [-1.091, -0.959, -0.677, -0.457, -0.372, -0.344, -0.271, -0.13 ,
         -0.149, -0.117, -0.02 , -0.043, -0.073],
        [-2.459, -2.073, -1.813, -1.626, -1.669, -1.556, -1.519, -1.484,
         -1.405, -1.338, -1.429, -1.21 , -1.228],
        [-3.822, -3.46 , -3.304, -2.981, -2.968, -2.844, -2.841, -2.84 ,
         -2.781, -2.669, -2.669, -2.595, -2.614]],

       [[ 0.361,  0.641,  0.888,  1.053,  1.149,  1.237,  1.308,  1.354,
          1.431,  1.469,  1.5  ,  1.551,  1.568],
        [ 0.342,  0.615,  0.868,  1.011,  1.111,  1.208,  1.26 ,  1.301,
          1.385,  1.429,  1.504,  1.531,  1.529],
        [-0.029,  0.264,  0.434,  0.527,  0.66 ,  0.796,  0.815,  0.876,
          0.949,  1.053,  1.06 ,  1.102,  1.108],
        [-1.158, -0.82 , -0.651, -0.566, -0.486, -0.307, -0.303, -0.155,
         -0.141, -0.116, -0.122, -0.017,  0.029],
        [-2.506, -2.21 , -1.959, -1.708, -1.524, -1.452, -1.471, -1.398,
         -1.266, -1.389, -1.304, -1.284, -1.185],
        [-3.9  , -3.597, -3.296, -3.099, -3.049, -2.989, -2.806, -2.666,
         -2.744, -2.724, -2.628, -2.531, -2.621]]])

for i in range(0, len(zeta_bins)):
    ratios = ratio_table[:, :, i]
    logratio = []

    for row in range(0, len(rho_bins)):
        for col in range(0, len(rs_bins)):
            logratio.append(ratios[row, col])

    interp_tables.append(interp2d(coords[:, 0], coords[:, 1], logratio))

def do_interp(lrho, lrs, lzeta):
    if isinstance(lzeta, float) or isinstance(lzeta, int):
        return do_interp_single(lrho, lrs, lzeta)[0]
    else:
        out = []
        for lzi in lzeta:
            out.append(do_interp_single(lrho, lrs, lzi))
        return np.array(out)


def do_interp_single(lrho, lrs, lzeta):
    lzbins = np.round(np.log10(zeta_bins), 2)

    sorted_inds = np.argsort(np.absolute(lzeta - lzbins))

    if lzeta <= lzbins[-1] and lzeta >= lzbins[0]:

        closest_ind = sorted_inds[0]
        second_closest_ind = sorted_inds[1]

        closest_zeta = lzbins[closest_ind]
        second_closest_zeta = lzbins[second_closest_ind]

        diff = np.absolute(second_closest_zeta - closest_zeta)

        weight1 = 1 - np.absolute(lzeta - closest_zeta) * diff ** -1
        weight2 = 1 - weight1
        # print(weight1, weight2)
        f1, f2 = interp_tables[closest_ind](lrho, lrs), interp_tables[second_closest_ind](lrho, lrs)
        return 10 ** (weight1 * f1 + weight2 * f2)

    elif lzeta > lzbins[-1]:

        last_zeta_midpoint = lzbins[-3]

        log_value_at_midpoint = np.log10(do_interp_single(lrho, lrs, last_zeta_midpoint))

        d_log_zeta = lzbins[-1] - last_zeta_midpoint

        derivative = (interp_tables[-1](lrho, lrs) - log_value_at_midpoint) * d_log_zeta ** -1

        intercept = interp_tables[-1](lrho, lrs)

        delta_zeta = lzeta - lzbins[-1]

        return 10 ** (derivative * delta_zeta + intercept)

    else:

        return do_interp_single(lrho, lrs, lzbins[0])

def halo_age(z, lookback_time_function, zform):

    formation_time = lookback_time_function(zform)
    age = lookback_time_function(z) - formation_time
    return np.max(age, 0)

def zeta_value(z, cross_0, lookback_time_function, zform):
    age = halo_age(z, lookback_time_function, zform)
    return age * cross_0

