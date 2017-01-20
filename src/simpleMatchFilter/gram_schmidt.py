"""
    Implmentation of Gram-Schmidt orthonormalization,
    also known as QR Factorization.
"""

import numpy as np


def gram_schmidt(A):
    """ Representation of Gram-Schmidt Process or QR Diagonalization
        for an mxn system of linear equations. """

    Q, R = np.linalg.qr(A)
    return Q, R


def __test():
    A = np.matrix('0 0 1 ; 0 1 0 ; 1 0 0 ')
    Q, R = gram_schmidt(A)
    print(Q)
    print(R)
    print(Q*R)

if __name__ == "__main__":
    __test()
