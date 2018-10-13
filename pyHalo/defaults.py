default_zstart = 0.01
distance_resolution_MPC = 1
default_z_round = 2
default_z_step = 0.025

default_mass_function = 'sheth99'

if default_mass_function == 'despali16':
    default_mdef = '200c'
if default_mass_function == 'reed07':
    default_mdef = 'fof'
if default_mass_function == 'sheth99':
    default_mdef = 'fof'

