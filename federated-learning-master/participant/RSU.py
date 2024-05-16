from time import time
from struct import pack

from models.func_w import decrypt
from random import getrandbits

from participant.Res_Sever import parameters, hash_1, hash_2, hash_4
from pypbc import *

pairing, g = parameters()


def get_exp(id):
    r = getrandbits(256)
    t = int(time())
    exp = str(id) + '&&&&&&&&' + str(t) + '&&&&&&&&' + str(r)
    return exp


class RSU:
    def __init__(self):
        self.id = 0
        self.y = 0  # rsu私钥
        self.pk = 0  # rsu公钥
        self.pkx = 0  # 主公钥
        self.pky = 0  # 验证公钥
        self.vehicle_st = dict()

    def register(self, arge):
        self.id = arge[0]
        self.y = arge[1]
        self.pk = arge[2]
        self.pkx = arge[3]
        self.pky = arge[4]
        self.vehicle_st = arge[5]

    def get_rsu(self):
        return self.id, self.pk

    def auth_without_token(self, req, EXP):
        # print('auth Ⅰ')
        id = req[0]
        TS = req[1]
        R1 = req[2]
        R2 = req[3]
        R3 = req[4]
        I = req[5]
        c = req[6]

        t1 = time()
        # 检测id是否有效
        if self.vehicle_st[id] == 0:
            return False
        t2 = time()
        print('check id and st:', t2 - t1)

        print('generate exp and communicate: unknown')

        t3 = time()
        K = hash_4(R3 ** self.y)
        t4 = time()
        print('compute K:', t4 - t3)

        t5 = time()
        w, bytew = decrypt(c, K)
        t6 = time()
        print('decrypt w and get bytes_w:', t6 - t5)

        t7 = time()
        # ID+TS+m
        m = id.to_bytes(8, byteorder='big') + pack('d', TS) + bytew
        t8 = time()
        print('generate str:', t8 - t7)

        t9 = time()
        left = pairing.apply(I, g)
        right = pairing.apply(R1, self.pkx) * pairing.apply(hash_2(m, R1), R2)
        t10 = time()
        print('authentication 1:', t10 - t9)

        if left != right:
            print('False')
            exit()

        t11 = time()
        sign = hash_1(EXP) ** self.y
        t12 = time()
        print('generate sign:', t12 - t11)

        T = [t2 - t1, t4 - t3, t6 - t5, t8 - t7, t10 - t9, t12 - t11]

        return {self.id: sign}, T, w

    def auth_with_token(self, req):
        # print('auth Ⅱ')

        id = req[0]
        exp = req[1]
        I = req[2]
        R3 = req[3]
        R4 = req[4]
        TS = req[5]
        c = req[6]

        # print('check id and exp: known')

        t1 = time()
        K = hash_4(R4 ** self.y)
        t2 = time()
        # print('compute K:', t2 - t1)

        t3 = time()
        w, bytew = decrypt(c, K)
        t4 = time()
        print('decrypt w and get bytes_w:', t4 - t3)

        t5 = time()
        m = id.to_bytes(8, byteorder='big') + pack('d', TS) + bytew
        t6 = time()
        # print('generate str:', t6 - t5)

        t7 = time()
        left = pairing.apply(I, g)
        right = pairing.apply(R3, self.pky) * pairing.apply(hash_2(m, R4 ** self.y), R4)
        t8 = time()
        print('authentication 2:', t8 - t7)

        if left != right:
            print('False')
            exit()

        T = [t2 - t1, t4 - t3, t6 - t5, t8 - t7]
        return T, w

    def authentication(self, req, is_token=True, exp=0):
        if not is_token:
            return self.auth_without_token(req, exp)
        elif is_token:
            return self.auth_with_token(req)
