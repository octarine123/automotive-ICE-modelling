"""
This module provides utility functions for processing, analyzing, and visualizing
engine simulation data.

Functions:
- dif_list(list_a): Calculate the absolute difference of each element from the initial value.
- eng_dict_report(user_dict): Generate a report of all key-value pairs in a dictionary.
- scan_dict(eng_dict, i, mode): Print specific engine parameters at a given index.
- create_plot(y_val, y_label, name): Create and save a plot of data against crank angle.
- plot_all(): Generate plots for all parameters specified in the global parameter dictionary.
- create_graphs(): Wrapper to generate all plots (calls plot_all).

Usage:
- Ensure that `param_dict` and `eng_dict` are properly initialized before calling functions
  that depend on them (e.g., `plot_all()`).
- The functions are designed to work with specific data structures representing engine parameters.
- Save plots will be stored in the 'outputs/graphs/' directory, which will be created if absent.

Note:
- This module assumes the presence of constants in `src.constants`
(e.g., `c.THETA_MIN`, `c.THETA_MAX`).
- Import this module and initialize data structures accordingly before usage.
"""
#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
import src.constants as c
from src.setup import param_dict, eng_dict



def dif_list(list_a):
    """
    Calculate absolute difference of each element from the initial value.
    First element is 0.
    """
    if not list_a:
        return []
    initial_value = list_a[0]
    result = [0]
    for item in list_a[1:]:
        diff = abs(item - initial_value)
        result.append(diff)
    return result


def eng_dict_report(user_dict):
    """Generate report on all sim values for checking."""
    if not isinstance(user_dict, dict):
        print("Provided argument is not a dictionary.")
        return
    print("Generating report on eng_dict:")
    for idx, key in enumerate(user_dict.keys()):
        value = user_dict[key]
        print(f"{idx}: Key = {key}, Value = {value}")


def scan_dict(data_dict, i, mode):
    """
     Extracts specific engine parameters at index i and optionally prints them.
     Parameters are retrieved from the provided dictionary.
     Mode 1 triggers printing of the parameters.
     """
    p_1 = data_dict["P1"][i]
    m_1 = data_dict["m_1"][i]
    t_1 = data_dict["t_1"][i]
    v_1 = data_dict["v_1"][i]
    if mode == 1:
        print(f"P1:{p_1} Pa, v_1:{v_1} m3, t_1:{t_1} K, m_1: {m_1}kg")



def create_plot(y_val, y_label, name):
    """
    create and save plot
    """
    # Ensure the directory exists
    dir_path = "outputs/graphs/"
    os.makedirs(dir_path, exist_ok=True)

    # Generate x_val with same length as y_val
    x_length = len(y_val)
    x_val = np.linspace(c.THETA_MIN, c.THETA_MAX, x_length)

    plt.plot(x_val, y_val)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(5, 4))
    plt.xlabel("crank angle (deg)")
    plt.ylabel(str(y_label), labelpad=2)
    plt.grid()
    filename = os.path.join(dir_path, f"{name}.png")
    plt.savefig(filename)
    plt.close()
    print(f"plot saved as {filename}.")


def plot_all():
    """
       Generate and save plots for all parameters in param_dict.
       Skips 'theta' and missing data, trims longer data to match 'theta'.
       Calls create_plot for each valid parameter and reports progress.
       """
    for i, j in zip(param_dict["param"], param_dict["units"]):
        if i == 'theta':
            continue
        y_vals = eng_dict.get(i, [])
        # Check if y_vals exists and is not empty
        if not y_vals:
            print(f"Warning: No data for parameter '{i}'. Skipping.")
            continue
        # Trim y_vals if longer than theta
        if len(y_vals) > len(eng_dict.get("theta", [])):
            y_vals = y_vals[:len(eng_dict["theta"])]
        y_label = f"{i} ({j})"
        create_plot(y_vals, y_label, i)
    print("All plots created.")


def create_graphs():
    """
       Generate all engine parameter plots.
       Calls plot_all to create and save each plot.
       Serves as a simple interface for visualization.
    """
    plot_all()
