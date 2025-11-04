import constants as c
import setup
import engine as eng


def enthalpy_rate(mass_flow, temp):
    """Calc enthalpy rate (J/s)"""
    delta_h = c.C_P * temp * mass_flow
    return delta_h


def enthalpy_calc(temp, m_dot_in, m_dot_ex):
    """calculate enthalpy change"""
    h_in = float(enthalpy_rate(m_dot_in, temp))
    h_ex = float(enthalpy_rate(m_dot_ex, temp))
    setup.eng_dict['H in'].append(h_in)
    setup.eng_dict['H ex'].append(h_ex)


def heat_transfer(p_in, h_pos, temp):
    """Calc q_out heat out lost from cylinder (# J/s)"""
    a_wall = 2 * eng.A_PISTON * setup.BORE * (eng.L_TDC + h_pos)
    a_sur = 2 * eng.A_PISTON + a_wall
    k_val = 6.194E-3 + 7.3814E-5 * temp - 1.2491E-8 * temp ** 2
    mu_val = 7.457E-6 + 4.1547E-8 * temp - 7.4793E-12 * temp ** 2
    s_len = 2 * setup.STROKE * (eng.RPM / 60)
    rho_gas = p_in / (temp * c.R_AIR)
    coef_heat = 0.49 * (k_val / setup.BORE) * ((rho_gas * setup.BORE * s_len) / mu_val) ** 0.7
    heat_trans = a_sur * coef_heat * (c.TEMP_WALL - temp)
    q_out = float(heat_trans * eng.DELTA_T)
    return q_out


def e_update(q_in, q_out, h_in, h_out):
    q_total = q_in + q_out + h_in + h_out
    return q_total
