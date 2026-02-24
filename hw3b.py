#region imports
from math import gamma, sqrt, pi
# We only need Simpson here. Ensure NumericalMethods.py is updated first!
from NumericalMethods import Simpson
#endregion
def t_distribution_PDF(args):
    """
    The integrand for the t-distribution.
    args = (u, m) where u is variable, m is degrees of freedom.
    """
    u, m = args
    return (1 + (u**2 / m))**(-(m + 1) / 2)

def main():
    print("---t-Distribution Probability Calculator---")
    # 1. User Inputs
    # Using 'int' for degrees of freedom is safer
    m = int(input("Degrees of freedom (m)? "))
    z = float(input("Value of z? "))

    # 2. Calculate Km (The constant part of the PDF)
    numerator = gamma((m + 1) / 2)
    denominator = sqrt(m * pi) * gamma(m / 2)
    km = numerator / denominator

    # 3. Integrate using Simpson
    lhl = -10.0  # Approximate negative infinity for t-dist
    rhl = z

    # We pass 'm' as a single-element tuple
    simpson_args = (m,)

    # Call the new generic Simpson function
    integral_value = Simpson(t_distribution_PDF, lhl, rhl, N=1000, args=simpson_args)

    probability = km * integral_value
    print(f"\nFor m = {m} and z = {z}:")
    print(f"Km = {km:.4f}")
    print(f"F(z) = {probability:.4f}")


if __name__ == "__main__":
    main()