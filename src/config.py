#!/usr/bin/env python
# coding: utf-8

with open("src/engine_config.json", "r") as file:
    config = json.load(file)

# operating conditions
P_ATM = 101.1 * 10 ** 3  # Pa
P_INLET_BOOST = -60.6 * 10 ** 3  # Pa
TEMP_INLET = 310.0  # degK
P_BACK_PRESSURE = 1.0 * 10 ** 3  # Pa
TEMP_ATM = 293.0  # degK
TEMP_EXHAUST = 878.0  # degK
P_INTAKE = 40.4 * 10 ** 3  # Pa
R_AIR = 287.0  # Jkg^-1K^-1
C_P = 1005.0  # Jkg^-1K^-1
C_v = 718.0  # Jkg^-1K^-1
GAM = 1.40
RHO_AIR = 1.225     # kg/m^3
TEMP_WALL = 473.0  # degK
P_CRANK_CASE = 100.0 * 10 ** 3  # Pa
GAS_A_FRI = 1 * 10 ** 5  # Pa
GAS_B_FRI = 250.0
THETA_MIN = -360.0  # deg
THETA_DELTA = 1.0 # deg
THETA_MAX = 360.0  # deg
AFR = 14.7
EFF_COMB = 0.91	# %
CAL_VAL = 42.500    # MJ/kg
CD = 0.65


A_CONROD = config["engine"]["area_conrod"]
AFR = config["combustion"]["afr"]
BORE = config["engine"]["bore"]
C_P = config['gas_properties']["Cp"]
CAL_VAL = config["combustion"]["calorific_value"]
COMP_RATIO = config["engine"]["compression_ratio"]
DELTA_T = (1 / (6 * RPM))  # /s to /deg
GAMMA = config['gas_properties']['gamma']
GAS_A_FRI = config["combustion"]["gasoline_friction_a"]
GAS_B_FRI = config["combustion"]["gasoline_friction_b"]
LEN_CONROD = config["engine"]["con_rod_length"]
MASS_CON_ROD = config["engine"]["mass_conrod"]
MASS_PISTON = config["engine"]["mass_piston"]
NUM_CYL = config["engine"]["num_cylinders"]
P_ATM = config['operating_conditions']['p_atm']
P_BACK_PRESSURE = config['operating_conditions']['p_back_pressure']
P_CRANK_CASE = config["simulation"]["p_crank_case"]
P_INTAKE = config["operating_conditions"]['p_intake']
R_AIR = config['gas_properties']['R_air']
RHO_AIR = config["gas_properties"]["rho_air"]
RPM = config["operating_conditions"]["rpm"]  # rpm"
STROKE = config["engine"]["stroke"]
TEMP_ATM = config["operating_conditions"]["temp_atm"]
TEMP_INLET = config["operating_conditions"]['temp_inlet']"
VEL_AVE = 2 * STROKE * (RPM / 60)
