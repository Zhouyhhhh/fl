import copy
from cryptography.fernet import Fernet
import numpy as np
from time import time


# 梯度加密
def encrypt(w, key):
    Bytew = b''
    en = []
    F = Fernet(key)
    for k in w:
        if type(k) == int and k == 0:
            en.append([0, 0, 0])
        else:
            shape = k.shape
            bytew = k.tobytes()
            Bytew += bytew
            enc = F.encrypt(bytew)
            en.append([enc, shape, k.dtype.name])

    return en, Bytew


# 梯度解密
def decrypt(w, key):
    Bytew = b''
    de = []
    F = Fernet(key)
    for W, shape, dtype in w:
        if type(shape) == int and shape == 0:
            dec = 0
        else:
            bytew = F.decrypt(W)
            Bytew += bytew
            if len(shape) == 1:
                dec = np.frombuffer(bytew, dtype=dtype).reshape(shape[0])
            elif len(shape) == 2:
                dec = np.frombuffer(bytew, dtype=dtype).reshape(shape[0], shape[1])
            elif len(shape) == 3:
                dec = np.frombuffer(bytew, dtype=dtype).reshape(shape[0], shape[1], shape[2])
            elif len(shape) == 4:
                dec = np.frombuffer(bytew, dtype=dtype).reshape(shape[0], shape[1], shape[2], shape[3])
        de.append(dec)
    return de, Bytew
