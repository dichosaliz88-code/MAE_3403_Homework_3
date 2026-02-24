# region imports
from NumericalMethods import GPDF, Probability, Secant


# endregion

# region function definitions
def main():
    """
    Integrates the Gaussian PDF from (mean-5stDev) to (c).
    """
    # 1. Decide mean, stDev, and c
    # Note: Using mu and sig to match your input names
    mu = float(input("Population mean (mu)? "))
    sig = float(input("Standard deviation (sig)? "))
    c = float(input("c value? "))

    # Check if we want P(x > c)
    is_GT = input("Probability greater than c? (y/n): ").lower() in ["y", "yes", "true"]

    # 2. Define args tuple
    # These must match the names mu and sig defined above
    args = (mu, sig)

    # 3. Call Probability
    # This sends (GPDF, (mu, sig), c, True/False)
    prob_val = Probability(GPDF, args, c, GT=is_GT)

    operator = ">" if is_GT else "<"

    # Print the result using the correct variable names
    print(f"P(x {operator} {c} | {mu}, {sig}) = {prob_val:.4f}")


if __name__ == "__main__":
    main()