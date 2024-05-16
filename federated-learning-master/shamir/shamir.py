from pypbc import *
from random import randrange

p = 730750818665451621361119245571504901405976559617


def invert(a):
    x0, x1, y0, y1 = 1, 0, 0, 1
    P = p
    while P != 0:
        q, a, P = a // P, P, a % P
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return x0 % p


# (t,n)门限的秘密共享，秘密为s
# id=[id1,id2,id3...]
def shamir_encryp(t, s, id, pairing):
    if t > len(id):
        return False
    a = []  # 系数
    share = dict()  # 分享份额
    for i in range(t - 1):
        aa = randrange(1, p - 1)
        a.append(aa)
    for i in id:
        ss = s
        for j in range(t - 1):
            zr = Element(pairing, Zr, value=int(int(i) ** (j + 1)) * a[j])
            ss += zr
        share[i] = Element(pairing, Zr, value=int(ss))
    return share


# f=f(i),t为最小门限
def shamir_decrypt(shares, g, pairing, t=2):
    if t != len(shares):
        return False
    sums = Element.zero(pairing, G1)
    for xj, yj in shares.items():
        prod = 1
        for xl, _ in shares.items():
            if xl != xj:
                a = xl
                b = xl - xj
                prod *= int(a * invert(b))
        prod = Element(pairing, Zr, value=prod)
        prod = Element(pairing, G1, value=yj ** prod)
        sums += prod

    return sums
