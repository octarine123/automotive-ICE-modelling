import numpy as np
import matplotlib.pyplot as plt
import constants as c
from setup import param_dict, eng_dict


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


def eng_dict_report():
    """generate report on all sim values for checking"""
    print("generating report")
    for k, i in enumerate(eng_dict.keys()):
        print(f"{k} = {i}")


def scan_dict(i, mode):
    p_1 = eng_dict["P1"][i]
    m_1 = eng_dict["m_1"][i]
    t_1 = eng_dict["t_1"][i]
    v_1 = eng_dict["v_1"][i]
    if mode == 1:
        print(f"P1:{p_1} Pa, v_1:{v_1} m3, t_1:{t_1} K, m_1: {m_1}kg")


def create_plot(y_val, y_label, name):
    """
    create and save plot
    """
    x_val = np.arange(c.THETA_MIN, c.THETA_MAX + 1, c.THETA_DELTA)
    plt.plot(x_val, y_val)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(5, 4))
    plt.xlabel("crank angle (deg)")
    plt.ylabel(str(y_label), labelpad=2)
    plt.grid()
    filename = str("outputs/graphs/" + name + ".png")
    plt.savefig(filename)
    plt.close()
    print(f"plot saved as {filename}.")


def plot_all():
    for i, j in zip(param_dict["param"], param_dict["units"]):
        if i == 'theta':
            continue
        y_vals = eng_dict[i]
        if len(y_vals) > len(eng_dict["theta"]):
            y_vals = y_vals[:-1]
        y_label = str(i + "(" + j + ")")
        create_plot(y_vals, y_label, i)
    print("all plots created.")


def create_graphs():
    plot_all()
