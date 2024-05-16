import numpy as np


def beaver_generate(n):
    a = np.random.rand(2 * n)
    d = a * a
    beaver = np.concatenate((a, d), axis=0)
    beaver1 = np.random.rand(4 * n)
    beaver2 = beaver - beaver1
    return beaver1, beaver2


def beaver_check(beaver1, beaver2):
    if len(beaver1) % 4 != 0:
        print("beaver wrong")
        return False
    n = int(len(beaver1) / 4)
    a01, a02, d01, d02 = beaver1[:n], beaver1[n:2 * n], beaver1[2 * n:3 * n], beaver1[3 * n:]
    a11, a12, d11, d12 = beaver2[:n], beaver2[n:2 * n], beaver2[2 * n:3 * n], beaver2[3 * n:]
    t = np.random.rand(n)
    e = t * (a01 + a11) - (a02 + a12)
    # S0 do
    c0 = t * t * d01 - d02 - 2 * t * e * a01 + e * e
    # S1 do
    c1 = t * t * d11 - d12 - 2 * t * e * a11
    c = np.around(c0 + c1, decimals=12)
    if np.any(c):
        print("beaver wrong")
        return False
    return True