#region imports
from NumericalMethods import GaussSeidel
#endregion

def main():
    #Problem 1
    #Augmented matrix
    Aaug1 = [
        [1, -1, 3, 2, 15],
        [-1, 5, -5, -2, -35],
        [3, -5, 19, 3, 94],
        [2, -2, 3, 21, 1]
    ]
    x1_guess = [0, 0, 0, 0]

    sol1 = GaussSeidel(Aaug1, x1_guess, Niter=50)

    #problem 2
    # Note: 0 is used where a variable is missing
    Aaug2 = [
        [4, 2, 4, 0, 20],
        [2, 2, 3, 2, 36],
        [4, 3, 6, 3, 60],
        [0, 2, 3, 9, 122]
    ]
    x2_guess = [0, 0, 0, 0]
    sol2 = GaussSeidel(Aaug2, x2_guess,Niter=50)

    print("Numerical Method Used: Gauss-Seidel Iteration")
    print("-" * 45)

    print("solution for problem 1:")
    print(f"x = [{', '.join([f'{val:.4f}' for val in sol1])}]")

    print("solution for problem 2:")
    print(f"x = [{', '.join([f'{val:.4f}' for val in sol2])}]")

if __name__ == "__main__":
    main()