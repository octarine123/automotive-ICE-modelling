#!/usr/bin/env python3
# coding: utf-8
# Engine Simulated: Nissan 1.8 Litre Petrol
# v2 to include more user input
"""
Internal Combustion Engine performance simulator
"""

print("ICE simulator started.")

import os
import math
import json
import numpy as np
from scipy.interpolate import barycentric_interpolate
from src.setup import theta_list, eng_dict, param_list, unit_list
import src.utils
import src.engine as eng
from src.gas_dynamics import flow_logic_intake, flow_logic_exhaust, m_dot_i_all_2, m_dot_e_all_2
from src.thermodynamics import heat_transfer
from src.combustion import comb_calcs
import pytest

with open("src/engine_config.json", "r") as file:
    config = json.load(file)

def find_nearest(array, value) -> float:
    idx = np.searchsorted(array, value, side='left')
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) <=
                    math.fabs(value - array[idx])):
        nearest = array[idx - 1]
    else:
        nearest = array[idx]
    return nearest


def shift_list(list_a, shift):
    """
    shifts list elements down and places elements shifted from bottom to top
    """
    if shift > len(list_a):
        shift = len(list_a)
    temp_a = list_a[shift:]
    temp_b = list_a[:shift]
    list_b = temp_a + temp_b
    return list_b


def temp_gas(pres, vol_gas, mass):
    """calc temp of gas (°K)"""
    temp = (pres * vol_gas) / (mass * R_AIR)
    return temp


def enthalpy_rate(mass_flow, temp):
    """Calc enthalpy rate (J/s)"""
    delta_h = C_P * temp * mass_flow
    return delta_h


def calc_circumference(rad):
    """
    circumference
    """
    circumference = math.pi * 2 * rad
    return circumference

GAMMA = config['gas_properties']['gamma']
P_ATM = config['operating_conditions']['p_atm']
P_BACK_PRESSURE = config['operating_conditions']['p_back_pressure']
R_AIR = config['gas_properties']['R_air']
C_P = config['gas_properties']["Cp"]
TEMP_INLET = config["operating_conditions"]['temp_inlet']
P_INTAKE = config["operating_conditions"]['p_intake']


piston_pos = {'TDC': [-360, 0],
              'BDC': [-180, 180]}

valve_data = {'num_valves': [2, 2],
              'diam_valves': [0.03, 0.022],
              'lift_peak': [0.0087, 0.0082],
              'cam_duration': [258.0, 260.0],  # CA deg
              'peak_lift_angle': [121.0, 115.0],  # b/atdc
              'flow_coeff': [0.65, 0.65],
              'A in': [],
              'A ex': []}

# cam data
I_LIFT = piston_pos['TDC'][0] + valve_data['peak_lift_angle'][0]
E_LIFT = 360 - valve_data['peak_lift_angle'][1]
I_OPEN = I_LIFT - valve_data['cam_duration'][0] / 2
I_CLOSE = I_OPEN + valve_data['cam_duration'][0]
E_OPEN = E_LIFT - valve_data['cam_duration'][1] / 2
E_CLOSE = E_OPEN + valve_data['cam_duration'][1]

# Create paths
root_path = os.getcwd() + "/"

# calc_constants
GAM_FUNC = (GAMMA ** 0.5) * (2 / (GAMMA + 1)) ** ((GAMMA + 1) / (2 * (GAMMA - 1)))
P_EXHAUST = P_ATM + P_BACK_PRESSURE
M_FUEL = 0.0


param_dict = {"param": param_list,
              "units": unit_list}


def cam_profile_gen_3(x_pos, y_pos, z_pos):
    x_observed = [x_pos, y_pos, z_pos]
    y_observed = [0, 1, 0]
    x_pos = np.arange(min(x_observed), max(x_observed))
    y_pos = barycentric_interpolate(x_observed, y_observed, x_pos)
    temp = np.array([x_pos, y_pos])
    # check for x_pos > 360 or -360 and shift values
    for x_pos in np.nditer(temp[0], op_flags=['readwrite']):
        if x_pos > 360:
            x_pos[...] = x_pos - 720
        if x_pos < -360:
            x_pos[...] = x_pos + 720
    cam_profile_list = np.array([np.array(np.sort(temp[0])),
                                 temp[1][np.argsort(temp[0])]])
    cam_profile_list = np.transpose(cam_profile_list)
    y_list = []
    for x_pos in theta_list:
        if abs(x_pos - find_nearest(cam_profile_list[:, 0], x_pos)) <= 1:
            i = np.where(cam_profile_list[:, 0] ==
                         find_nearest(cam_profile_list[:, 0], x_pos))
            value = float(cam_profile_list[i, 1])
            y_list.append(value)
        else:
            value = 0
            y_list.append(value)
    return y_list


def update_p2(q_tot, p_1, v_1, v_2):
    gam_val = (GAMMA + 1) / (2 * (GAMMA - 1))
    a_1 = (v_1 * gam_val - v_2 / 2)
    b_1 = (v_2 * gam_val - (v_1 / 2))
    p_2 = (q_tot + (p_1 * a_1)) / b_1
    eng_dict['P2'].append(p_2)
    eng_dict['P1'].append(p_2)
    return p_2


def m2_calc(m_1, m_i, m_e):
    """calc mass from mass flows (kg)"""
    m_2 = float(m_1 + (m_i + m_e))
    eng_dict['M2'].append(m_2)
    eng_dict['M1'].append(m_2)


def update_t2(p_2, v_2, m_2):
    """Calc T2, end step temperature"""
    t_2 = temp_gas(p_2, v_2, m_2)
    eng_dict['T2'].append(t_2)
    eng_dict['T1'].append(t_2)


# Mass initial conditions
def enthalpy_calc(temp, m_dot_in, m_dot_ex):
    """calculate enthalpy change"""
    h_in = float(enthalpy_rate(m_dot_in, temp))
    h_ex = float(enthalpy_rate(m_dot_ex, temp))
    eng_dict['H in'].append(h_in)
    eng_dict['H ex'].append(h_ex)


def e_update(q_in, q_out, h_in, h_out):
    q_total = q_in + q_out + h_in + h_out
    return q_total



def work_done():
    p_1 = np.array(eng_dict["P1"])[:-1]
    p_2 = np.array(eng_dict["P2"])
    v_1 = np.array(eng_dict["V1"])
    v_2 = np.array(eng_dict["V2"])
    delta_p = (p_1 + p_2) / 2
    delta_v = v_2 - v_1
    wd_val = list(np.multiply(delta_p, delta_v))
    return wd_val


def pre_calc():
    """ pre-calc initial conditions"""
    print("starting pre_calc")
    global M_FUEL
    theta_np = np.array(eng_dict['theta'])
    vol_1 = list(eng.v_cyl(theta_np))
    vol_2 = shift_list(vol_1, 1)
    eng_dict['V1'] = vol_1
    eng_dict['V2'] = vol_2
    h_1 = list(eng.len_h(theta_np))
    eng_dict["h"] = h_1
    i_calc = list(eng.i_total(theta_np))
    eng_dict["I total"] = i_calc
    eng_dict['M1'].append(eng.M_AIR_CYC)
    eng_dict["T1"].append(TEMP_INLET)
    eng_dict["P1"].append(P_INTAKE)
    # valve calc
    valve_i_circ = calc_circumference(valve_data['diam_valves'][0])
    valve_i_pk = valve_data['lift_peak'][0]
    list_a = list(np.array(cam_profile_gen_3(I_OPEN, I_LIFT, I_CLOSE))
                  * valve_i_circ * valve_i_pk)
    valve_data['A in'] = list_a
    valve_e_circ = calc_circumference(valve_data['diam_valves'][0])
    valve_i_pk = valve_data['lift_peak'][0]
    valve_data['A in'] = list(np.array(cam_profile_gen_3(I_OPEN,
                                                         I_LIFT,
                                                         I_CLOSE)) *
                              valve_e_circ * valve_i_pk)
    valve_e_pk = valve_data['lift_peak'][1]
    valve_data['A ex'] = list(np.array(cam_profile_gen_3(E_OPEN,
                                                         E_LIFT,
                                                         E_CLOSE)) *
                              valve_e_circ * valve_e_pk)
    # combustion energy data
    M_FUEL = eng.calc_m_fuel(eng.RPM, valve_data["cam_duration"][0])
    eng_dict["Q in"] = comb_calcs()
    print("finishing pre_calc")


def core_sim():
    """core sim loop"""
    print("core sim started")
    a_in = valve_data['A in']
    a_out = valve_data['A ex']
    for cnt, val in enumerate(eng_dict['theta']):
        flow_logic_i = flow_logic_intake(eng_dict["P1"][cnt])
        flow_logic_e = flow_logic_exhaust(eng_dict["P1"][cnt])
        m_f_i = m_dot_i_all_2(eng_dict['P1'][cnt],
                              eng_dict["T1"][cnt], a_in[cnt], flow_logic_i)
        m_f_e = m_dot_e_all_2(eng_dict['P1'][cnt],
                              eng_dict["T1"][cnt], a_out[cnt], flow_logic_e)
        eng_dict['m_dot_intake'].append(m_f_i)
        eng_dict['m_dot_exhaust'].append(m_f_e)
        m2_calc(eng_dict['M1'][cnt], m_f_i, m_f_e)
        # enthalpy and energy
        enthalpy_calc(eng_dict['T1'][cnt], m_f_i, m_f_e)
        q_out = heat_transfer(eng_dict["P1"][cnt],
                              eng_dict["h"][cnt], eng_dict["T1"][cnt])
        eng_dict['Q out'].append(q_out)
        q_tot = e_update(eng_dict["Q in"][cnt],
                         q_out, eng_dict["H in"][cnt], eng_dict["H ex"][cnt])
        eng_dict['Q total'].append(q_tot)
        update_p2(q_tot, eng_dict['P1'][cnt],
                  eng_dict['V1'][cnt], eng_dict['V2'][cnt])
        update_t2(eng_dict['P2'][cnt], eng_dict['V2'][cnt], eng_dict['M2'][cnt])
    # end of loop calc
    eng_dict["work done"] = work_done()
    eng_dict["P force"] = list(eng.p_force(np.array(eng_dict["P2"])))
    eng_dict["F comb"] = list(eng.f_comb(np.array(eng_dict["P force"]),
                                     eng_dict["I total"]))
    eng_dict["Conrod stress"] = list(eng.conrod_stress(np.array(
        eng_dict["F comb"])))
    print("core sim finished")


def main():
    pre_calc()
    core_sim()
    src.utils.plot_all()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as error:
        print('Program aborted.')
        raise SystemExit from error
    print("sim completed")
