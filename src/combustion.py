import constants as c
import math
from setup import theta_list
from utils import dif_list
import engine as eng

# Combustion Data
T_IGNITION = 16.0     # btdc_deg
BURN_DELAY = 6.0    # CA_deg
T_BURN = 48.00      # CA_deg
AFR = c.AFR
CAL_VAL = c.CAL_VAL
WEIBE_A = 5.0
M_FUEL = 0.0
WB_B = 2.0

# comb const
COMB_I = -T_IGNITION + BURN_DELAY    # atdc
THETA_COMB_F = COMB_I + T_BURN    # atdc


def gen_burn_logic(theta):
    """generate boolean list for burn period"""
    burn = bool(THETA_COMB_F >= theta > COMB_I)
    return burn


def calc_e_fuel(rpm, t_cam):
    """find total fuel energy for rpm (J)"""
    time = (t_cam / (rpm * 360 / 60))
    e_fuel = time * (M_FUEL * (CAL_VAL * 10 ** 6) * c.EFF_COMB) * (rpm / 120)
    return e_fuel


def comb_curve(theta, burn, e_fuel):
    """"produce cumulative combustion curve"""
    e_f = e_fuel
    if burn is True:
        comb_energy = (1 - math.exp(- WEIBE_A * ((theta - COMB_I) / T_BURN) **
                                    (WB_B + 1))) * e_f
    else:
        comb_energy = 0
    return comb_energy


def comb_calcs():
    """produce combustion curve data"""
    burn_list = []
    comb_list = []
    e_fuel = calc_e_fuel(eng.RPM, eng.valve_data["cam_duration"][0])
    for theta in theta_list:
        burn_list.append(gen_burn_logic(theta))
    for theta, burn in zip(theta_list, burn_list):
        e_comb = comb_curve(theta, burn, e_fuel)
        comb_list.append(e_comb)
    comb_int = dif_list(comb_list)
    return comb_int
