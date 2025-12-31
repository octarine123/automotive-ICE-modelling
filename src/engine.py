import numpy as np
import json
import src.setup
import src.constants as c
from src.setup import eng_dict, eval_dict, MASS_CON_ROD
import src.combustion

with open("src/engine_config.json", "r") as file:
    config = json.load(file)

BORE = config["engine"]["stroke"]
STROKE = config["engine"]["stroke"]
LEN_CONROD = config["engine"]["con_rod_length"]
NUM_CYL = config["engine"]["num_cylinders"]
COMP_RATIO = config["engine"]["compression_ratio"]
MASS_PISTON = config["engine"]["mass_piston"]
MASS_CON_ROD = config["engine"]["mass_conrod"]
P_CRANK_CASE = config["simulation"]["p_crank_case"]
RPM = 2500.0  # rpm
DELTA_T = (1 / (6 * RPM))  # /s to /deg
VEL_AVE = 2 * STROKE * (RPM / 60)
P_ATM = config["operating_conditions"]["p_atm"]
A_CONROD = config[][]

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
I_LIFT = piston_pos['TDC'][0] + valve_data['peak_lift_angle'][0]
E_LIFT = 360 - valve_data['peak_lift_angle'][1]
I_OPEN = I_LIFT - valve_data['cam_duration'][0] / 2
I_CLOSE = I_OPEN + valve_data['cam_duration'][0]
E_OPEN = E_LIFT - valve_data['cam_duration'][1] / 2
E_CLOSE = E_OPEN + valve_data['cam_duration'][1]

# mathematical formulae
def area_circle(diam):
    """
    area of circle(m2)
    """
    a_circle = float(np.pi * (diam / 2) ** 2)
    return a_circle


def eng_speed_rad(rpm):
    """convert eng speed from rpm to rad/s"""
    omega = rpm * ((2 * np.pi) / 60)
    return omega


def mass_gas(p_gas, vol_gas, temp):
    """
    calc mass of gas (kg)
    """
    mass = (p_gas * vol_gas) / (c.R_AIR * temp)
    return mass


A_PISTON = area_circle(BORE)  # m^2
SV = float(A_PISTON * STROKE)  # m^3
CV = SV / (COMP_RATIO - 1)  # m^3
MASS_OSC = MASS_PISTON + MASS_CON_ROD / 3  # kg
RADIUS_CRANK = STROKE / 2
M_AIR_IDEAL = mass_gas(P_ATM, SV, c.TEMP_ATM)
M_AIR_CYC = mass_gas(c.P_INTAKE, CV, c.TEMP_INLET)
eng_speed = eng_speed_rad(RPM)
L_TDC = CV / A_PISTON


def p_force(p_cyl):
    p_f_val = (p_cyl - P_CRANK_CASE) * A_PISTON
    return p_f_val


def len_h(theta):
    """
    calc piston positon (m)
    """
    len_a = np.sqrt(eng_dict.LEN_CONROD ** 2.0 -
                    (RADIUS_CRANK * np.sin(np.radians(theta))) ** 2)
    len_b = RADIUS_CRANK * np.cos(np.radians(theta))
    len_s = len_a + len_b
    len_h_sum = RADIUS_CRANK + eng_dict.LEN_CONROD - len_s
    return len_h_sum


def stress_conrod(theta):
    """calculate stress on piston (Pa)"""
    sigma = i_total(theta) / A_CON_ROD
    return sigma


def vel_piston(theta):
    """calc piston velocity (m/2)"""
    vel = - RADIUS_CRANK * eng_speed * (
                np.sin(np.radians(theta)) + 0.5 * (RADIUS_CRANK / LEN_CONROD) *
                np.sin(np.radians(2 * theta)))
    return vel


def accel_piston(theta):
    """calc piston accel (m/s2)"""
    accel = - RADIUS_CRANK * eng_speed ** 2 * (
                np.cos(np.radians(theta)) + (RADIUS_CRANK / LEN_CONROD) *
                np.cos(np.radians(2 * theta)))
    return accel


def v_cyl(theta):
    """calc piston volume (m^3)"""
    vol = A_PISTON * len_h(theta) + CV
    return vol


def fmep(rpm):
    """Friction mean effective pressure"""
    fmep_val = c.GAS_A_FRI + c.GAS_B_FRI * eng_dict.STROKE * rpm
    return fmep_val


def conrod_stress(force):
    stress = force / eng_dict.A_CON_ROD
    return stress

def imep(wc_gross_val):
    """Indicated Mean Effective Pressure"""
    imep_val = wc_gross_val / SV
    return imep_val


def pmep(wc_pump_val):
    """Pumping Mean Effective Pressure"""
    pmep_val = wc_pump_val / SV
    return pmep_val

def bmep(imep_val, fmep_val, pmep_val):
    """Brake Mean Effective Pressure"""
    bmep_val = imep_val - fmep_val - pmep_val
    return bmep_val


def wf_1_cyc():
    """Work one cycle (J)"""
    fmep_val = eval_dict["FMEP"]
    wf_one = -fmep_val * SV
    return wf_one


def pf_one_cyc():
    """Power 1 cycle (W)"""
    wf_one_val = eval_dict["Wf one"]
    pf_one_val = (wf_one_val * RPM) / 120
    return pf_one_val


def pf_total():
    """Total Power (W)"""
    pf_one = eval_dict["Pf one"]
    pf_total_val = pf_one * NUM_CYL
    return pf_total_val


def chem_eng():
    """Chemical Energy"""
    q_in = np.sum(np.array(eng_dict.eng_dict["Q in"]))
    chem_eng_val = (1 - c.EFF_COMB) * q_in
    return chem_eng_val


def heat_coolant():
    """Heat to Coolant (J)"""
    heat_loss = sum(src.setup.eng_dict["Q out"])
    wf_one = src.setup.eval_dict["Wf one"]
    heat_coolant_val = wf_one + heat_loss
    return heat_coolant_val


def h_total():
    """ Total enthalpy"""
    h_total_val = sum(eng_dict["H in"]) + sum(eng_dict["H ex"])
    return h_total_val



def w_exhaust():
    """work exhaust"""
    start = eng_dict["theta"].index(181)  # start exhaust
    end = eng_dict["theta"].index(360)  # end exhaust
    w_exh = float(np.sum(np.array(eng_dict["work done"][start:end])))
    return w_exh

def wc_gross():
    """wc_gross (J)"""
    w_comp_val = eval_dict["W comp"]
    w_exp_val = eval_dict["W exp"]
    wc_gross_val = w_comp_val + w_exp_val
    return wc_gross_val

def calc_m_fuel(rpm, t_cam):
    time = (t_cam / (rpm * 360 / 60))
    m_air = time * (c.RHO_AIR * SV * rpm / 120) * 0.85   # kg/s
    m_fuel = m_air / c.AFR
    return m_fuel


def eff_vol():
    end = eng_dict["theta"].index(-90.0)
    r_mass = np.sum(np.array(eng_dict['m_dot_intake'][:end]))
    eff_vol_val = r_mass / M_AIR_IDEAL
    return eff_vol_val


def w_intake():
    """work intake"""
    start = eng_dict["theta"].index(-359)  # start intake
    end = eng_dict["theta"].index(-180)  # end intake
    w_int_arr = np.array(eng_dict["work done"][start:end])
    w_int = float(np.sum(w_int_arr))
    return w_int

def w_comp():
    """work compression"""
    start = eng_dict["theta"].index(-179.0)  # start compression
    end = eng_dict["theta"].index(0.0)  # end compression
    w_comp_arr = (np.array(eng_dict["work done"][start:end]))
    w_comp_val = float(np.sum(w_comp_arr))
    return w_comp_val


def w_brake(bmep_val):
    w_brake_val  = bmep_val * SV
    return w_brake_val


def w_exp():
    """work expansion"""
    start = eng_dict["theta"].index(1.0)  # start expansion
    end = eng_dict["theta"].index(180.0)  # end expansion
    w_exp_arr = np.array(eng_dict["work done"][start:end])
    w_exp_val = float(np.sum(w_exp_arr))
    return w_exp_val


def wc_pump(w_intake_val, w_exhaust_val):
    wc_pump_val = -1 * (w_intake_val + w_exhaust_val)
    return wc_pump_val



def wc_gross():
    """wc_gross (J)"""
    w_comp_val = eval_dict["W comp"]
    w_exp_val = eval_dict["W exp"]
    wc_gross_val = w_comp_val + w_exp_val
    return wc_gross_val


def i_total(theta):
    """calculate forces on piston (N)"""
    i_sum = - (MASS_OSC *
               RADIUS_CRANK *
               (eng_speed ** 2) * np.cos(np.radians(theta))) - \
            (MASS_OSC * RADIUS_CRANK * (eng_speed ** 2) * RADIUS_CRANK) / \
            (LEN_CONROD * np.cos(np.radians(2 * theta)))
    return i_sum


def p_total(w_brake_val, rpm):
    """Total Power (W)"""
    p_1_cyc = w_brake_val * (rpm / 120)
    p_total_val = NUM_CYL * p_1_cyc
    return p_total_val


def f_comb(p_cyl, inertia):
    inertia_array = np.array(inertia)
    f_comb_val = p_cyl + inertia_array
    return f_comb_val

def t_total(power):
    """Torque (Nm)"""
    t_total_val = power / eng_speed
    return t_total_val


def eff_mech(imep_val, bmep_val):
    eff_mech_val = abs(bmep_val / imep_val)
    return eff_mech_val


def eff_therm(w_brake_val):
    eff_therm_val = abs(w_brake_val / ((c.CAL_VAL * 10 ** 6) * src.combustion.M_FUEL))
    return eff_therm_val


def bsfc():
    """Brake-specific fuel consumption (kg/kWhr)"""
    p_total_val = (eval_dict["P total"] / 1000)
    fuel_cons_val = eval_dict["fuel cons"]
    bsfc_val = fuel_cons_val / p_total_val
    return bsfc_val

def fuel_cons():
    """Fuel Consumption (kg/hr)"""
    fuel_cons_val = NUM_CYL * ((RPM * 60) / 2) * src.combustion.M_FUEL
    return fuel_cons_val

def evaluate_all():
    eval_dict["W comp"] = w_comp()
    eval_dict["W exp"] = w_exp()
    eval_dict["W exhaust"] = w_exhaust()
    eval_dict["W intake"] = w_intake()
    eval_dict["Wc gross"] = wc_gross()
    eval_dict["Wc pumping"] = wc_pump(eval_dict["W intake"],
                                      eval_dict["W exhaust"])
    eval_dict["IMEP"] = imep(eval_dict["Wc gross"])
    eval_dict["PMEP"] = (pmep(eval_dict["Wc pumping"]))
    eval_dict["FMEP"] = (fmep(RPM))
    eval_dict["BMEP"] = bmep(eval_dict["IMEP"],
                             eval_dict["FMEP"], eval_dict["PMEP"])
    eval_dict["W brake"] = w_brake(eval_dict["BMEP"])
    eval_dict["P total"] = p_total(eval_dict["W brake"], RPM)
    eval_dict["T total"] = t_total(eval_dict["P total"])
    eval_dict["eff mech"] = eff_mech(eval_dict["IMEP"], eval_dict["BMEP"])
    eval_dict["eff therm"] = eff_therm(eval_dict["W brake"])
    eval_dict["eff vol"] = eff_vol()
    eval_dict["fuel cons"] = fuel_cons()
    eval_dict["bsfc"] = bsfc()
    eval_dict["Wf one"] = wf_1_cyc()
    eval_dict["Pf one"] = pf_one_cyc()
    eval_dict["Pf total"] = pf_total()
    eval_dict["chem eng"] = chem_eng()
    eval_dict["heat coolant"] = heat_coolant()
    eval_dict["H total"] = h_total()
    print(eval_dict.values())