from src.utils import dif_list, eng_dict_report, scan_dict, create_plot, plot_all
import io
import numpy as np
import os
import sys
import glob


def test_dif_list():
    assert dif_list([1, 3, 6, 10]) == [0, 2, 5, 9]
    assert dif_list([10, 8, 5, 3]) == [0, 2, 5, 7]
    assert dif_list([0, 0, 1, 1]) == [0, 0, 1, 1]
    assert dif_list([]) == []
    assert dif_list([-1, -2, 0, 2]) == [0, 1, 1, 3]
    print("All tests passed!")


def test_eng_dict_report():
    # Sample dictionary for testing
    test_dict = {
        'alpha': 100,
        'beta': 200,
        'gamma': 300
    }

    # Capture the output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Call the function with test_dict
    eng_dict_report(test_dict)

    # Reset stdout
    sys.stdout = sys.__stdout__

    output = captured_output.getvalue()

    # Verify output contains expected lines
    assert "0: Key = alpha, Value = 100" in output
    assert "1: Key = beta, Value = 200" in output
    assert "2: Key = gamma, Value = 300" in output

    print("Test passed: eng_dict_report() outputs correct report.")


def test_eng_dict_report_non_dict():
    # Test with an argument that is not a dictionary
    non_dict_input = [1, 2, 3]  # list instead of dict

    # Capture output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Call the function
    eng_dict_report(non_dict_input)

    # Reset stdout
    sys.stdout = sys.__stdout__

    output = captured_output.getvalue()

    # Check if the correct message was printed
    assert "Provided argument is not a dictionary." in output

    print("Test passed: Non-dictionary input handled correctly.")


def test_scan_dict():
    global eng_dict
    # Sample data setup
    eng_dict = {
        "P1": [1000, 2000, 3000],
        "m_1": [0.5, 1.0, 1.5],
        "t_1": [300, 350, 400],
        "v_1": [0.01, 0.02, 0.03]
    }

    # Helper function to test mode 1 output
    def run_test(index, expected_output):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        scan_dict(eng_dict, index, 1)
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue().strip()
        assert expected_output in output, f"Expected to find '{expected_output}' in '{output}'"
        print(f"Passed for index {index}")

    # Test for valid indices
    run_test(0, "P1:1000 Pa, v_1:0.01 m3, t_1:300 K, m_1: 0.5kg")
    run_test(1, "P1:2000 Pa, v_1:0.02 m3, t_1:350 K, m_1: 1.0kg")
    run_test(2, "P1:3000 Pa, v_1:0.03 m3, t_1:400 K, m_1: 1.5kg")

    # Test for invalid index (out of range)
    try:
        scan_dict(eng_dict,3, 1)
        print("Failed: No exception raised for out-of-range index")
    except IndexError:
        print("Passed: IndexError raised for out-of-range index")
    except Exception as e:
        print(f"Failed: Unexpected exception {e}")

    # Optional: Test with mode other than 1 (should do nothing or handle differently)
    # For now, since mode !=1, no output, so we can test that
    captured_output = io.StringIO()
    sys.stdout = captured_output
    scan_dict(eng_dict,0, 0)  # mode !=1, no print expected
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert output == "", "Expected no output for mode != 1"
    print("Passed: No output for mode != 1")



def test_create_plot():
    # Sample data for testing
    import numpy as np
    y_val = np.array([1, 2, 3, 4, 5])
    y_label = "Test Y Label"
    name = "test_plot"

    # Call the function
    create_plot(y_val, y_label, name)

    # Path to the saved file
    filename = f"outputs/graphs/{name}.png"

    # Check if the file exists
    if os.path.exists(filename):
        print(f"Test passed: Plot file created at {filename}")
    else:
        print(f"Test failed: Plot file not found at {filename}")

    # Optional: Clean up the created file after test
    try:
        os.remove(filename)
        print("Cleanup: Removed test plot file.")
    except Exception as e:
        print(f"Cleanup failed: {e}")


# def test_plot_all():
#     # Setup mock data
#     global param_dict, eng_dict
#
#     param_dict = {
#         "param": ["theta", "param1", "param2"],
#         "units": ["deg", "unit1", "unit2"]
#     }
#
#     # Create dummy eng_dict with matching lengths
#     theta_length = 10
#     eng_dict = {
#         "theta": np.linspace(0, 180, theta_length),
#         "param1": np.random.rand(theta_length + 5),  # longer than theta
#         "param2": np.random.rand(theta_length)
#     }
#
#     # Ensure output directory exists
#     os.makedirs("outputs/graphs/", exist_ok=True)
#
#     # Call plot_all()
#     plot_all()
#
#     # Verify that plot images are created
#     for param in param_dict["param"]:
#         if param == "theta":
#             continue
#         filename = f"outputs/graphs/{param}.png"
#         if os.path.exists(filename):
#             print(f"PASS: {filename} exists.")
#         else:
#             print(f"FAIL: {filename} does not exist.")
#
#     # Optional: Clean up generated files
#     for param in param_dict["param"]:
#         filepath = f"outputs/graphs/{param}.png"
#         if os.path.exists(filepath):
#             os.remove(filepath)
#             print(f"Cleaned up {filepath}")


def test_plot_all():
    global param_dict, eng_dict

    # Define param_dict globally
    param_dict = {
        "param": ["theta", "param1", "param2"],
        "units": ["deg", "unit1", "unit2"]
    }
    print(param_dict)
    # Create dummy eng_dict
    theta_length = 10
    eng_dict = {
        "theta": np.linspace(0, 180, theta_length),
        "param1": np.random.rand(theta_length),
        "param2": np.random.rand(theta_length)
    }

    # Create output directory
    os.makedirs("outputs/graphs/", exist_ok=True)

    # Call plot_all()
    plot_all()

    # Check files
    for param in param_dict["param"]:
        if param == "theta":
            continue
        filename = f"outputs/graphs/{param}.png"
        if os.path.exists(filename):
            print(f"PASS: {filename} exists.")
        else:
            print(f"FAIL: {filename} does not exist.")

    # Cleanup
    for param in param_dict["param"]:
        filepath = f"outputs/graphs/{param}.png"
        if os.path.exists(filepath):
            os.remove(filepath)