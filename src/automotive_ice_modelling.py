#!/usr/bin/env python
# coding: utf-8
# Engine Simulated: Nissan 1.8 Litre Petrol

import os
import math
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import barycentric_interpolate
import constants as c


# tools

def boot_routine():
    print("welcome to the internal combustion engine simulator.")
    try:
        os.mkdir('inputs')
    except FileExistsError:
        print('inputs directory already exists.')
    try:
        os.mkdir('outputs')
        os.mkdir('outputs/graphs')
    except FileExistsError:
        print('outputs directory already exists.')


def load_csv_list(filename):
    """
    load csv as list
    """
    csv_list = []
    print(f"loading {filename}")
    with open(filename, 'r') as data:
        csv_reader = csv.reader(data)
        for row in csv_reader:
            for cell in row:
                # csv_list.append(float(cell))
                csv_list.append(cell)
    return csv_list


def save_dict_json(dict_1, file_name):
    full_name = str(file_name + ".json")
    with open(full_name, 'w') as f:
        json.dump(dict_1, f)
        print(f"dictionary saved as {full_name}.")


def save_dict_csv(dict_1, file_name):
    full_name = str(file_name + ".csv")
    my_dictionary = dict_1
    with open(full_name, 'w') as f:
        for key in my_dictionary.keys():
            f.write("%s, %s\n" % (key, my_dictionary[key]))
    f.close()
    print(f"dictionary saved as {full_name}.")


def load_list(file_name):
    """
    load file as list
    """
    print(f"loading {file_name}")
    file = open(root_path + file_name, 'r')
    content = file.read()
    content_list = content.split("\n")
    return content_list


def save_dict_to_file(dict_1, title):
    """
    save dictionary as txt file
    """
    file = open(title + '.txt', 'w')
    file.write(str(dict_1))
    file.close()
    print("data saved as " + title + '.txt')


def create_plot(y_val, y_label, name):
    """
    create and save plot
    """
    y_len = len(y_val)
    x_val = np.arange(THETA_MIN, THETA_MAX + 1, THETA_DELTA)
    plt.plot(x_val, y_val)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(5, 4))
    plt.xlabel("crank angle (deg)")
    plt.ylabel(str(y_label), labelpad=2)
    plt.grid()
    filename = str("outputs/graphs/" + name + ".png")
    plt.savefig(filename)
    plt.close()
    print(f"plot saved as {filename}.")


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side='left')
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) <
                    math.fabs(value - array[idx])):
        return array[idx - 1]
    else:
        return array[idx]


def shift_list(list_a, shift):
    """
    shifts list elements down and places elements shifted from botttom to top
    """
    if shift > len(list_a):
        shift = len(list_a)
    temp_a = list_a[shift:]
    temp_b = list_a[:shift]
    list_b = temp_a + temp_b
    return list_b


def angle_mod(theta):
    if theta > 360:
        theta = theta - 720
    if theta < -360:
        theta = theta + 720
    else:
        theta = theta
    return theta


def dif_list(list_a):
    """
    find difference between initial value and previous value in list
    """
    pre_val = 0
    list_b = []
    for item in list_a:
        dif_b = item - pre_val
        dif_b = max(dif_b, 0)
        list_b.append(dif_b)
        pre_val = item
    return list_b


def percentage_dif(val_a, val_b):
    dif = str(round(((val_a - val_b) / val_a) * 100, 1))
    dif = dif + '%'
    return dif


def compare_dicts(dict_a, dict_b):
    for val_a, val_b, key_a, key_b in zip(dict_a.values(),
                                          dict_b.values(),
                                          dict_a.keys(),
                                          dict_b.keys()):
        dif = percentage_dif(val_a, val_b)
        print(key_a, val_a, key_b, val_b, " :", dif)


# mathematical formulae
def area_circle(diam):
    """
    area of circle(m2)
    """
    a_circle = float(np.pi * (diam / 2) ** 2)
    return a_circle


def mass_gas(p_gas, vol_gas, temp):
    """
    calc mass of gas (kg)
    """
    mass = (p_gas * vol_gas) / (R_air * temp)
    return mass


def temp_gas(pres, vol_gas, mass):
    """
    calc temp of gas (°K)
    """
    temp = (pres * vol_gas) / (mass * R_air)
    return temp


def enthalpy_rate(mass_flow, temp):
    """
    Calc enthalpy rate (J/s)
    """
    delta_h = Cp * temp * mass_flow
    return delta_h


def eng_speed_rad(rpm):
    """
    convert eng speed from rpm to rad/s
    """
    omega = rpm * ((2 * np.pi) / 60)
    return omega


def calc_circ(rad):
    """
    circumference
    """
    circ = math.pi * 2 * rad
    return circ


def calc_m_fuel(rpm, t_cam):
    time = (t_cam / (rpm * 360 / 60))
    m_air = time * (rho_air * SV * rpm / 120) * 0.85   # kg/s
    m_fuel = m_air / AFR
    return m_fuel


# engine parameters
BORE = 0.080  # m
STROKE = 0.088  # m
LEN_CONROD = 0.1406  # m
COMP_RATIO = 9.62
NUM_CYL = 4
MASS_PISTON = 0.324  # kg
MASS_CONROD = 0.493  # kg
A_CONROD = 0.000152  # m^2
SIGMA_FAT_CONROD = 200 * 10 ** 6  # Pa
DIAM_BEARING = 0.05  # m
DIAM_BIG_END = 0.04  # m

# Combustion Data
T_IGNITITION = 16.0     # btdc deg
burn_delay = 6.0    # CAdeg
t_burn = 48.00      # CAdeg
AFR = c.AFR
Cal_Val = c.Cal_Val
WEIBE_A = 5.0
WB_m = 2.0

# operating conditions
ENG_RPM = 2500.0  # rpm
P_atm = c.P_atm
P_inlet_boost = c.P_inlet_boost
temp_inlet = c.temp_inlet
P_back_pressure = c.P_back_pressure
temp_atm = c.temp_atm
temp_exhaust = c.temp_exhaust
p_intake = c.p_intake
R_air = c.R_air
Cp = c.Cp
C_v = c.C_v
gam = c.gam
rho_air = c.rho_air
TEMP_WALL = c.TEMP_WALL
P_CRANK_CASE = c.P_CRANK_CASE
gasoline_a_friction = c.gasoline_a_friction
gasoline_b_friction = c.gasoline_b_friction
THETA_MIN = c.THETA_MIN
THETA_DELTA = c.THETA_DELTA
THETA_MAX = c.THETA_MAX
CD = c.CD
EFF_COMB = c.EFF_COMB

eng_speed = eng_speed_rad(ENG_RPM)

piston_pos = {'TDC': [-360, 0],
              'BDC': [-180, 180]}

valve_data = {'num_valves': [2, 2],
              'diam_valves': [0.03, 0.022],
              'lift_peak': [0.0087, 0.0082],
              'cam_duration': [258.0, 260.0],  # CAdeg
              'peak_lift_angle': [121.0, 115.0],  # b/atdc
              'flow_coeff': [0.65, 0.65],
              'A in': [],
              'A ex': []}

# cam data
i_lift = piston_pos['TDC'][0] + valve_data['peak_lift_angle'][0]
e_lift = 360 - valve_data['peak_lift_angle'][1]
i_open = i_lift - valve_data['cam_duration'][0] / 2
i_close = i_open + valve_data['cam_duration'][0]
e_open = e_lift - valve_data['cam_duration'][1] / 2
e_close = e_open + valve_data['cam_duration'][1]

# Create paths
root_path = os.getcwd() + "/"

# calc_constants
A_piston = area_circle(BORE)  # m^2
SV = float(A_piston * STROKE)  # m^3
CV = SV / (COMP_RATIO - 1)  # m^3
mass_osc = MASS_PISTON + MASS_CONROD / 3  # kg
PR_crit = (2 / (gam + 1)) ** (gam / (gam - 1))
vel_ave = 2 * STROKE * (ENG_RPM / 60)
mass_air_ideal = mass_gas(P_atm, SV, temp_atm)
gam_func = (gam ** 0.5) * (2 / (gam + 1)) ** ((gam + 1) / (2 * (gam - 1)))
delta_t = (1 / (6 * ENG_RPM))  # /s to /deg
P_exhaust = P_atm + P_back_pressure
M_FUEL = 0.0
RADIUS_CRANK = STROKE / 2

TV = SV + CV  # m^3
L_tdc = CV / A_piston  # m
mass_air_cyl = mass_gas(p_intake, CV, temp_inlet)

# comb const
comb_i = -T_IGNITITION + burn_delay    # atdc
theta_comb_f = comb_i + t_burn    # atdc

# setup
theta_list = list(np.arange(THETA_MIN, THETA_MAX + 1, THETA_DELTA))

eng_dict = {"theta": theta_list,
            "h": [],
            "I total": [],
            "M1": [],
            "m_dot_intake": [],
            "m_dot_exhaust": [],
            "M2": [],
            "P1": [],
            "P2": [],
            "V1": [],
            "V2": [],
            "T1": [],
            "T2": [],
            "H in": [],
            "H ex": [],
            "Q in": [],
            "Q out": [],
            "Q total": [],
            "P force": [],
            "F comb": [],
            "work done": [],
            "Conrod stress": []}

param_list = ['theta',
              'h',
              'I total',
              'M1',
              'm_dot_intake',
              'm_dot_exhaust',
              'M2',
              'P1',
              'P2',
              'V1',
              'V2',
              'T1',
              'T2',
              'H in',
              'H ex',
              'Q in',
              'Q out',
              'Q total',
              'P force',
              'F comb',
              'work done',
              'Conrod total stress']

unit_list = ['deg',
             'm',
             'N',
             'kg',
             'kg/s',
             'kg/s',
             'kg',
             'Pa',
             'Pa',
             'm3',
             'm3',
             'deg K',
             'deg K',
             'J',
             'J',
             'J',
             'J',
             'J',
             'N',
             'N',
             'J',
             'Pa']

param_dict = {"param": param_list,
              "units": unit_list}

dict_test_data = {}

eval_dict = {"FMEP": 0.0,
             "IMEP": 0.0,
             "BMEP": 0.0,
             "Wc gross": 0.0,
             "W comp": 0.0,
             "W exp": 0.0,
             "Wc pumping": 0.0,
             "W intake": 0.0,
             "W exhaust": 0.0,
             "PMEP": 0.0,
             "W brake": 0.0,
             "Pone cyl": 0.0,
             "P total": 0.0,
             "T total": 0.0,
             "eff mech": 0.0,
             "eff therm": 0.0,
             "eff vol": 0.0,
             "fuel cons": 0.0,
             "bsfc": 0.0,
             "Wf one": 0.0,
             "Pf one": 0.0,
             "Pf total": 0.0,
             "chem eng": 0.0,
             "heat coolant": 0.0,
             "H total": 0.0}


def scan_dict(i, mode):
    P1 = eng_dict["P1"][i]
    M1 = eng_dict["M1"][i]
    T1 = eng_dict["T1"][i]
    V1 = eng_dict["V1"][i]
    if mode == 1:
        print(f"P1:{P1} Pa, V1:{V1} m3, T1:{T1} K, M1: {M1}kg")


def len_h(theta):
    """
    calc piston positon (m)
    """
    len_a = np.sqrt(LEN_CONROD ** 2.0 -
                    (RADIUS_CRANK * np.sin(np.radians(theta))) ** 2)
    len_b = RADIUS_CRANK * np.cos(np.radians(theta))
    len_s = len_a + len_b
    len_h_sum = RADIUS_CRANK + LEN_CONROD - len_s
    return len_h_sum


def v_cyl(theta):
    """
    calc piston volume (m^3)
    """
    vol = A_piston * len_h(theta) + CV
    return vol


def vel_piston(theta):
    """
    calc piston velocity (m/2)
    """
    vel = - RADIUS_CRANK * eng_speed * (
                np.sin(np.radians(theta)) + 0.5 * (RADIUS_CRANK / LEN_CONROD) *
                np.sin(np.radians(2 * theta)))
    return vel


def accel_piston(theta):
    """
    calc piston accel (m/s2)
    """
    accel = - RADIUS_CRANK * eng_speed ** 2 * (
                np.cos(np.radians(theta)) + (RADIUS_CRANK / LEN_CONROD) *
                np.cos(np.radians(2 * theta)))
    return accel


def i_total(theta):
    """
    calculate forces on piston (N)
    """
    i_sum = - (mass_osc *
               RADIUS_CRANK *
               (eng_speed ** 2) * np.cos(np.radians(theta))) - \
               (mass_osc * RADIUS_CRANK * (eng_speed ** 2) * RADIUS_CRANK) / \
               (LEN_CONROD * np.cos(np.radians(2 * theta)))
    return i_sum


def stress_conrod(theta):
    """
    calculate stress on piston (Pa)
    """
    sigma = i_total(theta) / A_CONROD
    return sigma


def cam_profile_gen_3(x, y, z):
    x_observed = [x, y, z]
    y_observed = [0, 1, 0]
    x = np.arange(min(x_observed), max(x_observed))
    y = barycentric_interpolate(x_observed, y_observed, x)
    temp = np.array([x, y])
    # check for x > 360 or -360 and shift values
    for x in np.nditer(temp[0], op_flags=['readwrite']):
        if x > 360:
            x[...] = x - 720
        if x < -360:
            x[...] = x + 720
    cam_profile_list = np.array([np.array(np.sort(temp[0])),
                                 temp[1][np.argsort(temp[0])]])
    cam_profile_list = np.transpose(cam_profile_list)
    y_list = []
    for x in theta_list:
        if abs(x - find_nearest(cam_profile_list[:, 0], x)) <= 1:
            i = np.where(cam_profile_list[:, 0] ==
                         find_nearest(cam_profile_list[:, 0], x))
            value = float(cam_profile_list[i, 1])
            y_list.append(value)
        else:
            value = 0
            y_list.append(value)
    return y_list


def update_P2(q_tot, P1, V1, V2):
    gam_val = (gam + 1) / (2 * (gam - 1))
    a_1 = (V1 * gam_val - V2 / 2)
    b_1 = (V2 * gam_val - (V1 / 2))
    p_2 = (q_tot + (P1 * a_1)) / b_1
    eng_dict['P2'].append(p_2)
    eng_dict['P1'].append(p_2)
    return p_2


def intake_tests(p_cyl):
    """
    intake flow states
    """
    inflow = bool(p_cyl < p_intake)
    s_in = bool((p_cyl / p_intake) <= PR_crit)
    s_out = bool((p_intake / p_cyl) <= PR_crit)
    return inflow, s_in, s_out


def exhaust_dir_test(p_cyl):
    """
    calc exhaust gas flow direction
    """
    inflow = bool(p_cyl < P_exhaust)
    return inflow


def ex_in_s_test(p_cyl):
    sonic = bool((p_cyl / P_exhaust) <= PR_crit)
    return sonic


def ex_out_s_test(p_cyl):
    """
    exhaust sonic outflow test
    input: p_cyl (Pa)
    output: sonic (True/False)
    """
    sonic = bool((P_exhaust / p_cyl) <= PR_crit)
    return sonic


def eq_check(direction, inflow, outflow):
    """
    mass flow equation selection
    inputs: direction (True/False)
    inflow (True/False)
    outflow (True/False)
    output: tuple (True/False)
    """
    c_1 = bool(direction is True and inflow is False)
    c_2 = bool(direction is True and inflow is True)
    c_3 = bool(direction is False and outflow is False)
    c_4 = bool(direction is False and outflow is True)
    return c_1, c_2, c_3, c_4


# mass flow equations
def mass_flow_intake_c1(p_cyl, a_valve):
    """
    m dot for intake sub sonic inflow condition
    """
    mass_i_c1 = ((((CD * a_valve * p_intake) /
                   math.sqrt(R_air * temp_inlet)) *
                  (p_cyl / p_intake) ** (1 / gam)) *
                 ((2 * gam) / (gam - 1) *
                  (1 - (p_cyl / p_intake) ** ((gam - 1) / gam))) ** 0.5)
    return float(mass_i_c1.real)


def mass_flow_intake_c2(a_valve):
    """
    m dot for intake super sonic inflow condition
    """
    mass_i_c2 = ((CD * a_valve * p_intake) /
                 math.sqrt(R_air * temp_inlet)) * gam_func
    return float(mass_i_c2)


def mass_flow_intake_c3(p_cyl, a_valve, T_cyl):
    """
    m dot for intake sub sonic outflow condition
    """
    a = -(CD * a_valve * p_cyl) / \
    math.sqrt(R_air * T_cyl) * (p_intake / p_cyl) ** (1 / gam)
    b = ((2 * gam) / (gam - 1))
    c = (1 - (p_intake / p_cyl) ** ((gam - 1) / gam))
    mass_i_c3 = a * (b * c) ** 0.5
    return float(mass_i_c3)


def mass_flow_intake_c4(P_cyl, a_valve, T_cyl):
    """
    m dot for intake super sonic outflow condition
    """
    mass_i_c4 = -((CD * a_valve * P_cyl) /
                  math.sqrt(R_air * T_cyl)) * \
                  (gam ** 0.5) * (2 / (gam + 1)) ** \
                  ((gam + 1) / (2 * (gam - 1)))
    return mass_i_c4


def mass_flow_exhaust_c1(P_cyl, a_valve):
    """
    m dot for exhaust sub sonic inflow condition
    """
    a = (CD * a_valve * P_exhaust) / math.sqrt(R_air * temp_exhaust) * (P_cyl / P_exhaust) ** (1 / gam)
    b = (2 * gam) / (gam - 1)
    c = (1 - (P_cyl / P_exhaust) ** ((gam - 1) / gam))
    mass_e_c1 = a * b * c ** 0.5
    return mass_e_c1


def mass_flow_exhaust_c2(a_valve):
    """
    m dot for exhaust super sonic inflow condition
    """
    mass_e_c2 = ((CD * a_valve * P_exhaust) /
                 math.sqrt(R_air * temp_exhaust)) * gam ** 0.5 * (2 / (gam + 1)) ** ((gam + 1) / (2 * (gam - 1)))
    return mass_e_c2


def mass_flow_exhaust_c3(p_cyl, a_valve, T_cyl):
    """
    m dot for exhaust sub sonic outflow condition
    """
    mass_e_c3 = -((CD * a_valve * p_cyl) /
                  math.sqrt(R_air * T_cyl)) * (P_exhaust / p_cyl) ** (1 / gam) * (((2 * gam / (gam - 1)) * (1 - (P_exhaust / p_cyl) ** ((gam - 1) / gam)))) ** 0.5
    return mass_e_c3


def mass_flow_exhaust_c4(p_cyl, a_valve, T_cyl):
    """
    m dot for exhaust super sonic outflow condition
    inputs - p_cyl (Pa) a_valve (m2) T_cyl (deg K)
    outputs - m_dot (kg/s)
    """
    a = -(CD * a_valve * p_cyl) / math.sqrt(R_air * T_cyl)
    b = (gam ** 0.5)
    c = (2 / (gam + 1)) ** ((gam + 1) / (2 * (gam - 1)))
    mass_e_c4 = a * b * c
    return mass_e_c4


def flow_logic_intake(p_cyl):
    """
    intake valve mass flow eq logic
    input - p_cyl (Pa)
    output - True/False
    """
    logic_i = []
    i_tup = intake_tests(p_cyl)
    intake_dir = i_tup[0]
    inflow_sonic = i_tup[1]
    outflow_sonic = i_tup[2]
    logic_i = list(eq_check(intake_dir, inflow_sonic, outflow_sonic))
    return logic_i


def flow_logic_exhaust(p_cyl):
    """
    exhaust valve mass flow eq logic
    input - p_cyl (Pa)
    output - True/False
    """
    logic_e = []
    exhaust_dir = exhaust_dir_test(p_cyl)
    exhaust_inflow_sonic = ex_in_s_test(p_cyl)
    exhaust_outflow_sonic = ex_out_s_test(p_cyl)
    logic_e = list(eq_check(exhaust_dir,
                            exhaust_inflow_sonic, exhaust_outflow_sonic))
    return logic_e


def m_dot_i_all_2(p_cyl, temp, area, logic):
    """
    calc all m dot intake values
    inputs - p_cy (Pa), temp (deg K), area (m2)
    output - m_flow (kg/deg)
    """
    if logic[0] is True:
        m_flow_i = (mass_flow_intake_c1(p_cyl, area))
    elif logic[1] is True:
        m_flow_i = mass_flow_intake_c2(area)
    elif logic[2] is True:
        m_flow_i = mass_flow_intake_c3(p_cyl, area, temp)
    elif logic[3] is True:
        m_flow_i = mass_flow_intake_c4(p_cyl, area, temp)
    m_flow_i = m_flow_i * delta_t
    return m_flow_i


def m_dot_e_all_2(p_cyl, temp, area, logic):
    """
    calc all m dot exhaust values
    """
    if logic[0] is True:
        m_flow_e = (mass_flow_exhaust_c1(p_cyl, area))
    elif logic[1] is True:
        m_flow_e = mass_flow_exhaust_c2(area)
    elif logic[2] is True:
        m_flow_e = mass_flow_exhaust_c3(p_cyl, area, temp)
    elif logic[3] is True:
        m_flow_e = mass_flow_exhaust_c4(p_cyl, area, temp)
    m_flow_e = m_flow_e * delta_t
    return m_flow_e


def m2_calc(m_1, m_i, m_e):
    """
    calc mass from mass flows
    inputs: m_1 (kg), m_i (kg/deg), m_e (kg/deg)
    output: m_2 (kg)
    """
    m_2 = float(m_1 + (m_i + m_e))
    eng_dict['M2'].append(m_2)
    eng_dict['M1'].append(m_2)


def update_T2(P2, V2, M2):
    """Calc T2, end step temperature
Inputs:
*******
P2 : end step pressure (Pa)
V2: end step cylinder volume (m3)
M2: end step internal mass (kg)

Returns:
********
T2 : float, end step temperature (K)
"""
    T2 = temp_gas(P2, V2, M2)
    eng_dict['T2'].append(T2)
    eng_dict['T1'].append(T2)


# Mass initial conditions
def enthalpy_calc(temp, m_dot_in, m_dot_ex):
    """
    calculate ethalpy change
    """
    h_in = float(enthalpy_rate((m_dot_in), temp))
    h_ex = float(enthalpy_rate((m_dot_ex), temp))
    eng_dict['H in'].append(h_in)
    eng_dict['H ex'].append(h_ex)


def gen_burn_logic(theta):
    """
    generate bollean list for burn period
    """
    burn = bool(theta_comb_f >= theta > comb_i)
    return burn


def calc_e_fuel(rpm):
    """"
    find total fuel energy for rpm (J)
    """
    e_fuel = M_FUEL * (Cal_Val * 10 ** 6) * EFF_COMB
    return e_fuel


def comb_curve(theta, burn, e_fuel):
    """"
    produce cumlative combustion curve
    """
    pre_val = 0
    e_f = e_fuel
    if burn is True:
        comb_energy = (1 - math.exp(- WEIBE_A * ((theta - comb_i) / t_burn) **
                                    (WB_m + 1))) * e_f
        pre_val = comb_energy
    else:
        comb_energy = pre_val
    return comb_energy


def comb_calcs():
    """
    produce combustion curve data
    """
    burn_list = []
    comb_list = []
    e_fuel = calc_e_fuel(ENG_RPM)
    for theta in theta_list:
        burn_list.append(gen_burn_logic(theta))
    for theta, burn in zip(theta_list, burn_list):
        e_comb = comb_curve(theta, burn, e_fuel)
        comb_list.append(e_comb)
    comb_int = dif_list(comb_list)
    return comb_int


def heat_transfer(p_in, h_pos, temp):
    """
    Calc q_out heat out lost from cylinder (# J/s)
    """
    A_wall = 2 * A_piston * BORE * (L_tdc + h_pos)
    a_sur = 2 * A_piston + A_wall
    K = 6.194E-3 + 7.3814E-5 * temp - 1.2491E-8 * temp ** 2
    mu = 7.457E-6 + 4.1547E-8 * temp - 7.4793E-12 * temp ** 2
    S = 2 * STROKE * (ENG_RPM / 60)
    rho_gas = p_in / (temp * R_air)
    coef_heat = 0.49 * (K / BORE) * ((rho_gas * BORE * S) / mu) ** 0.7
    heat_trans = a_sur * coef_heat * (TEMP_WALL - temp)
    q_out = float(heat_trans * delta_t)
    return q_out


def e_update(q_in, q_out, h_in, h_out):
    q_total = q_in + q_out + h_in + h_out
    return q_total


def plot_all():
    for i, j in zip(param_dict["param"], param_dict["units"]):
        if i is 'theta':
            continue
        y_vals = eng_dict[i]
        if len(y_vals) > len(eng_dict["theta"]):
            y_vals = y_vals[:-1]
        y_label = str(i + "(" + j + ")")
        create_plot(y_vals, y_label, i)
    print("all plots created.")


"""
EVALUATION
"""


def FMEP(rpm):
    """Friction mean effective pressure
    INPUTS:
    stroke, gasoline friction a, gasoline friction b, rpm

    OUTPUTS:
    FMEP (Pa)
"""
    FMEP = gasoline_a_friction + gasoline_b_friction * STROKE * rpm
    return FMEP


def W_comp():
    """work compression"""
    start = eng_dict["theta"].index(-179.0)  # start compression
    end = eng_dict["theta"].index(0.0)  # end compression
    w_comp_arr = (np.array(eng_dict["work done"][start:end]))
    w_comp = float(np.sum(w_comp_arr))
    return w_comp


def Wc_gross():
    """Wc_gross
    INPUTS:
    W_comp (J), W_exp (J)
    OUTPUTS:
    Wc_gross (J)
    """
    W_comp = eval_dict["W comp"]
    W_exp = eval_dict["W exp"]
    Wc_gross = W_comp + W_exp
    return Wc_gross


def IMEP(Wc_gross):
    """IMEP
    Indicated Mean Effective Pressure
    INPUTS:
    Wc_gross (J)
    SV - swept volume (m3)

    OUTPUTS:
    IMEP (Pa)
    """
    IMEP = Wc_gross / SV
    return IMEP


def P_force(p_cyl):
    P_force = (p_cyl - P_CRANK_CASE) * A_piston
    return P_force


def F_comb(p_cyl, inertia):
    inertia_array = np.array(inertia)
    F_comb = p_cyl + inertia_array
    return F_comb


def work_done():
    p_1 = np.array(eng_dict["P1"])[:-1]
    p_2 = np.array(eng_dict["P2"])
    V_1 = np.array(eng_dict["V1"])
    V_2 = np.array(eng_dict["V2"])
    delta_P = (p_1 + p_2) / 2
    delta_V = V_2 - V_1
    WD = list(np.multiply(delta_P, delta_V))
    return WD


def conrod_stress(force):
    stress = force / A_CONROD
    return stress


def BMEP(IMEP, FMEP, PMEP):
    """
    Brake Mean Effective Pressure
    """
    BMEP = IMEP - FMEP - PMEP
    return BMEP


def PMEP(Wc_pump):
    """
    Pumping Mean Effective Pressure
    work done by piston/cycle in intake/exhaust strokes
    """
    PMEP = Wc_pump / SV
    return PMEP


def Wc_pump(w_intake, w_exhaust):
    wc_pump = -1 * (w_intake + w_exhaust)
    return wc_pump


def W_exp():
    """work expansion"""
    start = eng_dict["theta"].index(1.0)  # start expansion
    end = eng_dict["theta"].index(180.0)  # end expansion
    w_exp_arr = np.array(eng_dict["work done"][start:end])
    w_exp = float(np.sum(w_exp_arr))
    return w_exp


def W_exhaust():
    """work exhaust"""
    start = eng_dict["theta"].index(181)  # start exhaust
    end = eng_dict["theta"].index(360)  # end exhaust
    w_exh = float(np.sum(np.array(eng_dict["work done"][start:end])))
    return w_exh


def W_intake():
    """work intake"""
    start = eng_dict["theta"].index(-359)  # start intake
    end = eng_dict["theta"].index(-180)  # end intake
    w_int_arr = np.array(eng_dict["work done"][start:end])
    w_int = float(np.sum(w_int_arr))
    return w_int


def W_brake(BMEP):
    w_brake = BMEP * SV
    return w_brake


def P_total(W_brake, rpm):
    """
    Total Power (W)
    """
    P_one_cyc = W_brake * (rpm / 120)
    P_total = NUM_CYL * P_one_cyc
    return P_total


def T_total(power):
    """
    Torque
    inputs- power (W), engine speed (rad/s)
    output - Torque (Nm)
    """
    T_total = power / eng_speed
    return T_total


def eff_mech(IMEP, BMEP):
    eff_mech = abs(BMEP / IMEP)
    return eff_mech


def eff_therm(w_brake):
    eff_therm = abs(w_brake / ((Cal_Val * 10 ** 6) * M_FUEL))
    return eff_therm


def eff_vol():
    end = eng_dict["theta"].index(-90.0)
    r_mass = np.sum(np.array(eng_dict['m_dot_intake'][:end]))
    eff_vol = r_mass / mass_air_ideal
    return eff_vol


def fuel_cons():
    """Fuel Consumption
    inputs- # cyl, eng rpm, total mass fuel (kg)
    outputs- fuel consumption (kg/hr)
    """
    fuel_cons = NUM_CYL * ((ENG_RPM * 60) / 2) * M_FUEL
    return fuel_cons


def bsfc():
    """"
    Brake-specific fuel consumption
    inputs- fuel cons (kg/hr), Ptotal(W)
    outputs- bsfc (kg/kWhr)
    """
    P_total = (eval_dict["P total"] / 1000)
    fuel_cons = eval_dict["fuel cons"]
    bsfc = fuel_cons / P_total
    return bsfc


def Wf_one_cycle():
    """
    Work one cycle
##    inputs- FMEP (Pa), SV (m3)
    outputs- Wf (J)
    """
    FMEP = eval_dict["FMEP"]
    Wf_one = -FMEP * SV
    return Wf_one


def Pf_one_cycle():
    """
    Power 1 cycle
    inputs- work done (J)
    putput - power (W)
    """
    wf_one = eval_dict["Wf one"]
    Pf_one = (wf_one * ENG_RPM) / 120
    return Pf_one


def Pf_total():
    """
    Total Power (W)
    inputs - power one cycle (W)
    output - Pf total (W)
    """
    Pf_one = eval_dict["Pf one"]
    Pf_total = Pf_one * NUM_CYL
    return Pf_total


def chem_eng():
    """"
    Chemical Energy
    input- combustion eff (%), Q_in (J)
    outputs- chem_eng (J)
    """
    Q_in = np.sum(np.array(eng_dict["Q in"]))
    chem_eng = (1 - EFF_COMB) * Q_in
    return chem_eng


def heat_coolant():
    """
    Heat to Coolant (J)
    inputs- Wf_one (J), Heat loss (J)
    outputs - heat_coolant (J)
    """
    heat_loss = sum(eng_dict["Q out"])
    Wf_one = eval_dict["Wf one"]
    heat_coolant = Wf_one + heat_loss
    return heat_coolant


def H_total():
    """ Total enthalpy
    inputs- H_in, H_ex
    outputs- H_total (J)
    """
    H_total = sum(eng_dict["H in"]) + sum(eng_dict["H ex"])
    return H_total


def evaluate_all():
    eval_dict["W comp"] = W_comp()
    eval_dict["W exp"] = W_exp()
    eval_dict["W exhaust"] = W_exhaust()
    eval_dict["W intake"] = W_intake()
    eval_dict["Wc gross"] = Wc_gross()
    eval_dict["Wc pumping"] = Wc_pump(eval_dict["W intake"],
                                      eval_dict["W exhaust"])
    eval_dict["IMEP"] = IMEP(eval_dict["Wc gross"])
    eval_dict["PMEP"] = (PMEP(eval_dict["Wc pumping"]))
    eval_dict["FMEP"] = (FMEP(ENG_RPM))
    eval_dict["BMEP"] = BMEP(eval_dict["IMEP"],
                             eval_dict["FMEP"], eval_dict["PMEP"])
    eval_dict["W brake"] = W_brake(eval_dict["BMEP"])
    eval_dict["P total"] = P_total(eval_dict["W brake"], ENG_RPM)
    eval_dict["T total"] = T_total(eval_dict["P total"])
    eval_dict["eff mech"] = eff_mech(eval_dict["IMEP"], eval_dict["BMEP"])
    eval_dict["eff therm"] = eff_therm(eval_dict["W brake"])
    eval_dict["eff vol"] = eff_vol()
    eval_dict["fuel cons"] = fuel_cons()
    eval_dict["bsfc"] = bsfc()
    eval_dict["Wf one"] = Wf_one_cycle()
    eval_dict["Pf one"] = Pf_one_cycle()
    eval_dict["Pf total"] = Pf_total()
    eval_dict["chem eng"] = chem_eng()
    eval_dict["heat coolant"] = heat_coolant()
    eval_dict["H total"] = H_total()


def eng_dict_report():
    """
    generate report on all sim values for checking
    """
    for k, i in enumerate(eng_dict.keys()):
        num = len(eng_dict[i])
        print(f"{k} = {i} = {num}")


def pre_calc():
    """
    pre-calc initial conditions
    """
    global M_FUEL
    theta_np = np.array(eng_dict['theta'])
    vol_1 = list(v_cyl(theta_np))
    vol_2 = shift_list(vol_1, 1)
    eng_dict['V1'] = vol_1
    eng_dict['V2'] = vol_2
    h_1 = list(len_h(theta_np))
    eng_dict["h"] = h_1
    I_calc = list(i_total(theta_np))
    eng_dict["I total"] = I_calc
    eng_dict['M1'].append(mass_air_cyl)
    eng_dict["T1"].append(temp_inlet)
    eng_dict["P1"].append(p_intake)
    # valve calc
    valve_i_circ = calc_circ(valve_data['diam_valves'][0])
    valve_i_pk = valve_data['lift_peak'][0]
    list_a = list(np.array(cam_profile_gen_3(i_open, i_lift, i_close))
                  * valve_i_circ * valve_i_pk)
    valve_data['A in'] = list_a
    valve_e_circ = calc_circ(valve_data['diam_valves'][0])
    valve_i_pk = valve_data['lift_peak'][0]
    valve_data['A in'] = list(np.array(cam_profile_gen_3(i_open,
                                                         i_lift,
                                                         i_close)) *
                              valve_e_circ * valve_i_pk)
    valve_e_pk = valve_data['lift_peak'][1]
    valve_data['A ex'] = list(np.array(cam_profile_gen_3(e_open,
                                                         e_lift,
                                                         e_close)) *
                              valve_e_circ * valve_e_pk)
    # combustion energy data
    M_FUEL = calc_m_fuel(ENG_RPM, valve_data["cam_duration"][0])
    eng_dict["Q in"] = comb_calcs()


def core_sim():
    """
    core sim loop
    """
    a_in = valve_data['A in']
    a_out = valve_data['A ex']
    for ct, val in enumerate(eng_dict['theta']):
        flow_logic_i = flow_logic_intake(eng_dict["P1"][ct])
        flow_logic_e = flow_logic_exhaust(eng_dict["P1"][ct])
        m_f_i = m_dot_i_all_2(eng_dict['P1'][ct],
                              eng_dict["T1"][ct], a_in[ct], flow_logic_i)
        m_f_e = m_dot_e_all_2(eng_dict['P1'][ct],
                              eng_dict["T1"][ct], a_out[ct], flow_logic_e)
        eng_dict['m_dot_intake'].append(m_f_i)
        eng_dict['m_dot_exhaust'].append(m_f_e)
        m2_calc(eng_dict['M1'][ct], m_f_i, m_f_e)
        # enthalpy and energy
        enthalpy_calc(eng_dict['T1'][ct], m_f_i, m_f_e)
        q_out = heat_transfer(eng_dict["P1"][ct],
                              eng_dict["h"][ct], eng_dict["T1"][ct])
        eng_dict['Q out'].append(q_out)
        q_tot = e_update(eng_dict["Q in"][ct],
                         q_out, eng_dict["H in"][ct], eng_dict["H ex"][ct])
        eng_dict['Q total'].append(q_tot)
        update_P2(q_tot, eng_dict['P1'][ct],
                  eng_dict['V1'][ct], eng_dict['V2'][ct])
        update_T2(eng_dict['P2'][ct], eng_dict['V2'][ct], eng_dict['M2'][ct])
    # end of loop calcs
    eng_dict["work done"] = work_done()
    eng_dict["P force"] = list(P_force(np.array(eng_dict["P2"])))
    eng_dict["F comb"] = list(F_comb(np.array(eng_dict["P force"]),
                                     eng_dict["I total"]))
    eng_dict["Conrod stress"] = list(conrod_stress(np.array(
        eng_dict["F comb"])))


def main():
    pre_calc()
    core_sim()
    eng_dict_report()


def quit_option(ans):
    print("do you wish to end the program?")


def cui(window):
    print(window.keys())
    command = input("> ")
    try:
        window[command]()
    except KeyError:
        print("invalid command, please try again")
        cui(window)


def create_graphs():
    plot_all()
    cui(main_menu)


def read_output():
    print("read output function")


def new_project():
    print("new project function")


def open_project():
    print("open project function")


intro_menu = {'new project': new_project,
              'open project': open_project,
              'end': quit}

# main menu to call functions
main_menu = {'run sim': main,
             'create graphs': create_graphs,
             'read output': read_output,
             'end': quit}


if __name__ == '__main__':
    boot_routine()
    try:
        cui(intro_menu)
        cui(main_menu)
    except KeyboardInterrupt:
        print('Program aborted.')
        raise SystemExit
    evaluate_all()
    save_dict_csv(eval_dict, "outputs/eng_evaluation_data")
    save_dict_json(eng_dict, "outputs/eng_sim_data")
    print("sim completed")
    cui(main_menu)
