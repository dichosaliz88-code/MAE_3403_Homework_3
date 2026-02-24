# region imports
import Gauss_Elim as GE
from math import sqrt, pi, exp, cos


# endregion

# region function definitions
def GPDF(args):
    """
    Gaussian probability density function.
    :param args: (x, mean, standard deviation)
    :return: value of GPDF at x
    """
    x, mu, sig = args
    return (1 / (sig * sqrt(2 * pi))) * exp(-0.5 * ((x - mu) / sig) ** 2)


def Simpson(fn, lhl, rhl, N=100, args=()):
    """
    Generic Simpson's 1/3 rule.
    :param lhl: Lower integration limit
    :param rhl: Upper integration limit
    :param args: Extra arguments to pass to fn (e.g., mu, sig OR degrees of freedom)
    """
    m = N if N % 2 == 0 else N + 1
    h = (rhl - lhl) / m

    # We construct the tuple (x, *args) to pass to the target function
    f_lhl = fn((lhl,) + args)
    f_rhl = fn((rhl,) + args)
    _Sum = f_lhl + f_rhl

    odd_sum = 0
    even_sum = 0

    for i in range(1, m):
        x = lhl + i * h
        fx = fn((x,) + args)
        if i % 2 == 1:
            odd_sum += fx
        else:
            even_sum += fx

    _Sum += (4 * odd_sum) + (2 * even_sum)
    return (h / 3) * _Sum


def Probability(PDF, args, c, GT=True):
    """
    Updated to use the generic Simpson function.
    """
    mu, sig = args
    lhl = mu - 5 * sig
    rhl = c

    # We pass (mu, sig) as the 'args' tuple to Simpson
    p = Simpson(PDF, lhl, rhl, args=(mu, sig))

    return 1 - p if GT else p


def Secant(fcn, x0, x1, maxiter=10, xtol=1e-5):
    """
    Implements Secant method for root finding.
    """
    iter = 0
    x_diff = abs(xtol) + 1
    while iter < maxiter and abs(x_diff) > xtol:
        f0 = fcn(x0)
        f1 = fcn(x1)
        if f1 - f0 == 0: break  # Avoid division by zero

        x_New = x1 - f1 * ((x1 - x0) / (f1 - f0))
        x_diff = x_New - x1
        x0 = x1
        x1 = x_New
        iter += 1
    return (x1, iter)


def GaussSeidel(Aaug, x, Niter=15):
    """
    Gauss-Seidel method for solving Ax=b.
    """
    Aaug = GE.MakeDiagDom(Aaug)
    n_Rows = len(Aaug)
    n_Cols = len(Aaug[0]) - 1
    for j in range(Niter):
        for i in range(n_Rows):
            rhs = Aaug[i][n_Cols]
            for k in range(n_Cols):
                if k != i:
                    rhs -= Aaug[i][k] * x[k]
            x[i] = rhs / Aaug[i][i]
    return x


def Transpose(A):
    """
    Matrix transpose utility.
    """
    if isinstance(A[0], list):
        n = len(A)
        m = len(A[0])
        if m == 1:
            return [x[0] for x in A]
        return [[A[j][i] for j in range(n)] for i in range(m)]
    else:
        return [[x] for x in A]


def main():
    # Test GPDF
    fx = GPDF((0, 0, 1))
    print(f"GPDF at 0: {fx:0.5f}")

    # Test Simpson (Integral from -5 to 0 should be 0.5)
    p = Simpson(GPDF, -5, 0, args=(0, 1))
    print(f"Simpson p: {p:0.5f}")

    # Test Probability
    p1 = Probability(GPDF, (0, 1), 0, True)
    print(f"p1 (x > 0): {p1:0.5f}")

    # Probability within 1, 2, and 3 standard deviations
    print(f"1-sigma: {1 - 2 * Probability(GPDF, (0, 1), 1, True):0.5f}")
    print(f"2-sigma: {1 - 2 * Probability(GPDF, (0, 1), 2, True):0.5f}")
    print(f"3-sigma: {1 - 2 * Probability(GPDF, (0, 1), 3, True):0.5f}")


if __name__ == '__main__':
    main()