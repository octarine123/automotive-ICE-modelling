import numpy as np
import constants as c
import csv


def open_csv_dict(filename):
    output_dict = {}
    print(f"loading {filename}")
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            for key, value in row.items():
                output_dict[key] = value
    csvfile.close()
    return output_dict


# setup
theta_list = list(np.arange(c.THETA_MIN, c.THETA_MAX + 1, c.THETA_DELTA))
theta_len = len(theta_list)

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
              'Conrod stress']

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

param_dict = open_csv_dict("inputs/engine_parameters_landscape.csv")

# engine parameters
BORE = float(param_dict['BORE'])
STROKE = param_dict['STROKE']
LEN_CONROD = param_dict['LEN_CONROD']
COMP_RATIO = param_dict['COMP_RATIO']
NUM_CYL = param_dict['NUM_CYL']
MASS_PISTON =  param_dict['MASS_PISTON']
MASS_CON_ROD =  param_dict['MASS_CONROD']
A_CON_ROD =  param_dict['A_CONROD']
SIGMA_FAT_CON_ROD =  param_dict['SIGMA_FAT_CONROD']
DIAM_BEARING =  param_dict['DIAM_BEARING']
DIAM_BIG_END =  param_dict['DIAM_BIG_END']

