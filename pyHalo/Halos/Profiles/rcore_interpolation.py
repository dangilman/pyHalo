import numpy as np
from scipy.interpolate import interp2d

linear_term = np.array([[-1.9375420256061098, -1.822312192119458, -1.720107843234162, -1.5239896613615465, -1.3244640489722208, -1.0923162240842579],
[-1.758064459389106, -1.4934807413277202, -1.372741443618585, -1.1235201531480186, -0.9281179411020659, -0.7531325173584733],
[-0.895521309353152, -0.6922464165614682, -0.6170578601649971, -0.509141360518114, -0.4206817197343252, -0.3273788009490688],
[-0.6327372792995511, -0.49549271857664107, -0.4511744854225804, -0.3775054976320997, -0.23600481756295116, -0.0745956682627925],
[-0.17095591713398192, -0.10134415001369539, -0.03711397821049154, -0.05027775938269236, -0.08940312658783671, -0.0840892375773059],
])

constant_term = np.array([[11.907105028032456, 11.148756159002266, 10.785011196625016, 10.19458693536923, 9.645438726726978, 9.101459084265091],
[11.217960296449085, 10.390856738612596, 10.060141647981968, 9.49665485013243, 9.02029723519018, 8.625851930669052],
[9.65072643019597, 9.141630138993474, 8.97221982294337, 8.71977783105231, 8.50733603539536, 8.313308330681265],
[9.315095553596707, 8.978866064990143, 8.852365969228856, 8.673913013075705, 8.47948717159852, 8.280114192566506],
[9.102789799556206, 8.983101296945845, 8.900133122921416, 8.899107212284008, 8.955710229754542, 8.94254224752026],
])

rho_bins = np.array([5 * 10 ** 6, 10 ** 7, 5 * 10 ** 7, 10 ** 8, 5 * 10 ** 8])
rs_bins = np.array(10 ** np.array([-0.75, -0.4, -0.25, 0.0, 0.25, 0.50]))
log_rs_arr, log_rho_arr = np.meshgrid(np.log10(rs_bins), np.log10(rho_bins))
linear_interp = interp2d(log_rho_arr, log_rs_arr, linear_term)
constant_interp = interp2d(log_rho_arr, log_rs_arr, constant_term)

def halo_age(z, lookback_time_function, zform = 10):

    formation_time = lookback_time_function(zform)
    age = lookback_time_function(z) - formation_time
    return np.max(age, 0)

def zeta_value(z, cross_0, lookback_time_function):
    age = halo_age(z, lookback_time_function)
    return age * cross_0

def interp_rc_over_rs(rho_s, Rs, zeta):

    log_rho_value = np.log10(rho_s)
    log_rs_value = np.log10(Rs)
    log_zeta_value = np.log10(zeta)

    p0 = linear_interp(log_rho_value, log_rs_value)
    p1 = constant_interp(log_rho_value, log_rs_value)

    rho_sidm = 10 ** (p0 * log_zeta_value + p1)
    return 10 ** log_rho_value * rho_sidm ** -1
