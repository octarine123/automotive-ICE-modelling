import math
import constants as c
import engine as eng

PR_CRT = (2 / (c.GAM + 1)) ** (c.GAM / (c.GAM - 1))
P_EXHAUST = c.P_ATM + c.P_BACK_PRESSURE
GAM_FUNC = (c.GAM ** 0.5) * (2 / (c.GAM + 1)) ** ((c.GAM + 1) / (2 * (c.GAM - 1)))


def intake_tests(p_cyl):
    """
    intake flow states
    """
    inflow = bool(p_cyl < c.P_INTAKE)
    s_in = bool((p_cyl / c.P_INTAKE) <= PR_CRT)
    s_out = bool((c.P_INTAKE / p_cyl) <= PR_CRT)
    return inflow, s_in, s_out


def exhaust_dir_test(p_cyl):
    """calc exhaust gas flow direction"""
    inflow = bool(p_cyl < P_EXHAUST)
    return inflow


def ex_in_s_test(p_cyl):
    sonic = bool((p_cyl / P_EXHAUST) <= PR_CRT)
    return sonic


def ex_out_s_test(p_cyl):
    """exhaust sonic outflow test (boolean)"""
    sonic = bool((P_EXHAUST / p_cyl) <= PR_CRT)
    return sonic


def eq_check(direction, inflow, outflow):
    """mass flow equation selection tuple (True/False)"""
    c_1 = bool(direction is True and inflow is False)
    c_2 = bool(direction is True and inflow is True)
    c_3 = bool(direction is False and outflow is False)
    c_4 = bool(direction is False and outflow is True)
    return c_1, c_2, c_3, c_4


# mass flow equations
def m_dot_i_c1(p_cyl, a_valve):
    """
    m dot for intake sub sonic inflow condition
    """
    mass_i_c1 = ((((c.CD * a_valve * c.P_INTAKE) /
                   math.sqrt(c.R_AIR * c.TEMP_INLET)) *
                  (p_cyl / c.P_INTAKE) ** (1 / c.GAM)) *
                 ((2 * c.GAM) / (c.GAM - 1) *
                  (1 - (p_cyl / c.P_INTAKE) ** ((c.GAM - 1) / c.GAM))) ** 0.5)
    return float(mass_i_c1.real)


def m_dot_i_c2(a_valve):
    """m dot for intake supersonic inflow condition"""
    mass_i_c2 = ((c.CD * a_valve * c.P_INTAKE) /
                 math.sqrt(c.R_AIR * c.TEMP_INLET)) * GAM_FUNC
    return float(mass_i_c2)


def m_dot_i_c3(p_cyl, a_valve, t_cyl):
    """m dot for intake sub sonic outflow condition"""
    elm_1 = -(c.CD * a_valve * p_cyl) / \
            math.sqrt(c.R_AIR * t_cyl) * (c.P_INTAKE / p_cyl) ** (1 / c.GAM)
    elm_2 = ((2 * c.GAM) / (c.GAM - 1))
    elm_3 = (1 - (c.P_INTAKE / p_cyl) ** ((c.GAM - 1) / c.GAM))
    mass_i_c3 = elm_1 * (elm_2 * elm_3) ** 0.5
    return float(mass_i_c3)


def m_dot_i_c4(p_cyl, a_valve, t_cyl):
    """m dot for intake supersonic outflow condition"""
    mass_i_c4 = -((c.CD * a_valve * p_cyl) /
                  math.sqrt(c.R_AIR * t_cyl)) * \
                (c.GAM ** 0.5) * (2 / (c.GAM + 1)) ** \
                ((c.GAM + 1) / (2 * (c.GAM - 1)))
    return mass_i_c4


def mass_flow_exhaust_c1(p_cyl, a_valve):
    """m dot for exhaust sub sonic inflow condition"""
    elm_1 = (c.CD * a_valve * P_EXHAUST) / math.sqrt(c.R_AIR * c.TEMP_EXHAUST) * \
            (p_cyl / P_EXHAUST) ** (1 / c.GAM)
    elm_2 = (2 * c.GAM) / (c.GAM - 1)
    elm_3 = (1 - (p_cyl / P_EXHAUST) ** ((c.GAM - 1) / c.GAM))
    mass_e_c1 = elm_1 * elm_2 * elm_3 ** 0.5
    return mass_e_c1


def mass_flow_exhaust_c2(a_valve):
    """m dot for exhaust supersonic inflow condition"""
    mass_e_c2 = ((c.CD * a_valve * P_EXHAUST) /
                 math.sqrt(c.R_AIR * c.TEMP_EXHAUST)) * c.GAM ** 0.5 * \
                (2 / (c.GAM + 1)) ** ((c.GAM + 1) / (2 * (c.GAM - 1)))
    return mass_e_c2


def mass_flow_exhaust_c3(p_cyl, a_valve, t_cyl):
    """m dot for exhaust sub sonic outflow condition (kg/s)"""
    mass_e_c3 = -((c.CD * a_valve * p_cyl) /
                  math.sqrt(c.R_AIR * t_cyl)) * (P_EXHAUST / p_cyl) ** \
                (1 / c.GAM) * (((2 * c.GAM / (c.GAM - 1)) * (1 - (P_EXHAUST / p_cyl) **
                                                             ((c.GAM - 1) / c.GAM)))) ** 0.5
    return mass_e_c3


def mass_flow_exhaust_c4(p_cyl, a_valve, t_cyl):
    """m dot for exhaust supersonic outflow condition (kg/s)"""
    elm_1 = -(c.CD * a_valve * p_cyl) / math.sqrt(c.R_AIR * t_cyl)
    elm_2 = (c.GAM ** 0.5)
    elm_3 = (2 / (c.GAM + 1)) ** ((c.GAM + 1) / (2 * (c.GAM - 1)))
    mass_e_c4 = elm_1 * elm_2 * elm_3
    return mass_e_c4


def flow_logic_intake(p_cyl):
    """intake valve mass flow eq logic (boolean)"""
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
    exhaust_dir = exhaust_dir_test(p_cyl)
    exhaust_inflow_sonic = ex_in_s_test(p_cyl)
    exhaust_outflow_sonic = ex_out_s_test(p_cyl)
    logic_e = list(eq_check(exhaust_dir,
                            exhaust_inflow_sonic, exhaust_outflow_sonic))
    return logic_e


def m_dot_i_all_2(p_cyl, temp, area, logic):
    """calc all m dot intake values (kg/deg)    """
    if logic[0] is True:
        m_flow_i = (m_dot_i_c1(p_cyl, area))
    elif logic[1] is True:
        m_flow_i = m_dot_i_c2(area)
    elif logic[2] is True:
        m_flow_i = m_dot_i_c3(p_cyl, area, temp)
    elif logic[3] is True:
        m_flow_i = m_dot_i_c4(p_cyl, area, temp)
    m_flow_i = m_flow_i * eng.DELTA_T
    return m_flow_i


def m_dot_e_all_2(p_cyl, temp, area, logic):
    """calc all m dot exhaust values"""
    if logic[0] is True:
        m_flow_e = (mass_flow_exhaust_c1(p_cyl, area))
    elif logic[1] is True:
        m_flow_e = mass_flow_exhaust_c2(area)
    elif logic[2] is True:
        m_flow_e = mass_flow_exhaust_c3(p_cyl, area, temp)
    elif logic[3] is True:
        m_flow_e = mass_flow_exhaust_c4(p_cyl, area, temp)
    m_flow_e = m_flow_e * eng.DELTA_T
    return m_flow_e
