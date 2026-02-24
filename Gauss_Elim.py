# region explanation
# This program is used to teach Gauss elimination by row operations
# the elementary operations for Gauss elimination are
# 1. Swap the positions of two rows (I'll use pop and insert)
# 2. Multiply a row by a non-zero scalar
# 3. Add to one row, a scalar multiple of another row
# endregion

# region imports
import copy as CP
from copy import deepcopy as dc  # a quick way to access deepcopy through an alias


# endregion

# region functions
def FirstNonZero_Index(R):
    """
    Finds pivot for a row (i.e., first non-zero number in a row reading from left to right)
    :param R: a row vector
    :return: the column index (start counting at zero) of the first non-zero number
    """
    for ColumnIndex in range(len(R)):
        if R[ColumnIndex] != 0.0:
            return ColumnIndex
    return -1


def MakeDiagDom(A):
    n = len(A)
    # This will hold our reordered rows in their correct positions
    reordered_A = [None] * n
    # Keep track of which original rows we've already placed
    rows_used = [False] * n

    for j in range(n):  # For each variable column j
        best_row = -1
        max_ratio = -1

        for i in range(n):  # Search through all rows i
            if not rows_used[i]:
                diag_val = abs(A[i][j])
                # Sum of other coefficients in the row
                off_diag_sum = sum(abs(A[i][k]) for k in range(n) if k != j)

                ratio = diag_val / off_diag_sum if off_diag_sum != 0 else 1e10

                if ratio > max_ratio:
                    max_ratio = ratio
                    best_row = i

        reordered_A[j] = A[best_row]
        rows_used[best_row] = True

    return reordered_A


# region row operations
def SwapRows(A, r1, r2):
    '''
    One of the elementary row operations in Gaussian elimination.
    '''
    rmax = max(r1, r2)
    rmin = min(r1, r2)
    RMax = A[rmax]
    RMin = A.pop(rmin)
    A.insert(rmin, RMax)
    A[rmax] = RMin
    return A


def MultRow(R, s=1):
    '''
    Used to multiply a row vector by a scalar value.
    '''
    for i in range(len(R)):
        R[i] *= s
    return R


def AddRows(R1, R2, s=1.0):
    '''
    Adds a scalar multiple of row vector R2 to row vector R1.
    '''
    RNew = CP.deepcopy(R1)
    for i in range(len(R1)):
        RNew[i] += R2[i] * s
    return RNew


# endregion

def EchelonForm(A):
    '''
    Gaussian elimination to produce echelon form matrix.
    '''
    m = len(A)
    n = len(A[0])
    Ech = CP.deepcopy(A)

    for i in range(m):
        for r in range(i, m):
            p = FirstNonZero_Index(Ech[r])
            if p == i:
                Ech = SwapRows(Ech, r, i)
                break
        #this is the "elimination" part
        if (Ech[i][i] != 0.0):
            for r in range(i + 1, m):
                p = FirstNonZero_Index(Ech[r])
                if p == i:
                    Row = Ech[r]
                    #this ratio determines how much the pivot row to subtract
                    s = -Ech[r][p] / Ech[i][i]
                    Ech[r] = AddRows(Row, Ech[i], s)
    return Ech


def ReducedEchelonForm(A):
    """
    Calculates reduced echelon form of A.
    """
    REF = EchelonForm(A)
    for i in range(len(A) - 1, -1, -1):
        R = REF[i]
        j = FirstNonZero_Index(R)
        if j != -1:
            R = MultRow(R, 1.0 / R[j])
            REF[i] = R
            for ii in range(i - 1, -1, -1):
                RR = REF[ii]
                if (RR[j] != 0):
                    RR = AddRows(RR, R, -RR[j])
                    REF[ii] = RR
    return REF


def IDMatrix(A):
    m = len(A)
    n = len(A[0])
    IM = [[1 if j == i else 0 for j in range(n)] for i in range(m)]
    return IM


def AugmentMatrix(A, B):
    C = CP.deepcopy(A)
    for i in range(len(C)):
        C[i].extend(B[i])
    return C


def popColumn(A, j):
    numRows = len(A)
    AA = dc(A)
    c = [0] * numRows
    for rowIndex in range(numRows):
        c[rowIndex] = AA[rowIndex].pop(j)
    return c, AA


def insertColumn(A, b, i):
    ANew = dc(A)
    for r in range(len(ANew)):
        newRow = dc(ANew[r])
        newRow.insert(i, b[r])
        ANew[r] = dc(newRow)
    return ANew


def InvertMatrix(A):
    """
    Finds the inverse of matrix A.
    """
    ID = IDMatrix(A)
    Ainv = AugmentMatrix(A, ID)
    IAinv = ReducedEchelonForm(Ainv)
    # Correctly pop columns to leave only the inverse
    for j in range(len(A[0]) - 1, -1, -1):
        _, IAinv = popColumn(IAinv, j)
    return IAinv


def MatrixMultiply(A, B):
    '''
    Standard matrix multiplication (m x n) * (n x p)
    '''
    m = len(A)
    n = len(A[0])
    p = len(B[0])

    # Check dimensions
    if n != len(B):
        return None

    C = [[0 for _ in range(p)] for _ in range(m)]

    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
            C[i][j] = round(C[i][j], 3)
    return C


def main():
    M = [[4, -1, -1, 3], [-2, -3, 1, 9], [-1, 1, 7, -6]]
    print("Original matrix:")
    for r in M: print(r)

    # pop last column to get A
    b_col, A = popColumn(M, len(M[0]) - 1)

    MI = InvertMatrix(A)
    print("\nInverted Matrix:")
    for r in MI: print(r)

    B = MatrixMultiply(A, MI)
    print("\nA * A^-1 (Identity Check):")
    for r in B: print(r)


if __name__ == "__main__":
    main()
