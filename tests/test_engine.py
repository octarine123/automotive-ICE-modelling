from src.engine import p_force, P_CRANK_CASE, A_PISTON


def test_p_force():
    # Define constants for testing
    global P_CRANK_CASE, A_PISTON
    # P_CRANK_CASE = 50  # example value
    # A_PISTON = 10  # example value
    print(P_CRANK_CASE, A_PISTON)

    # Test case 1
    p_cyl = 100
    expected_force = (p_cyl - P_CRANK_CASE) * A_PISTON  # (100 - 50)*10 = 500
    result = p_force(p_cyl)
    assert result == expected_force, f"Expected {expected_force}, got {result}"
    # Test case 2
    p_cyl = 75
    expected_force = (75 - P_CRANK_CASE) * A_PISTON
    result = p_force(p_cyl)
    assert result == expected_force, f"Expected {expected_force}, got {result}"
    print("All tests passed for p_force!")


# Run the test
test_p_force()