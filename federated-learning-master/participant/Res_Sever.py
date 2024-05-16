import base64
import hashlib

from pypbc import *
from random import getrandbits
from shamir.shamir import shamir_encryp, shamir_decrypt

stored_params = """type a
q 8780710799663312522437781984754049815806883199414208211028653399266475630880222957078625179422662221423155858769582317459277713367317481324925129998224791
h 12016012264891146079388821366740534204802954401251311822919615131047207289359704531102844802183906537786776
r 730750818665451621361119245571504901405976559617
exp2 159
exp1 107
sign1 1
sign0 1
"""

# initialize elliptic curve
params = Parameters(param_string=stored_params)
pairing = Pairing(params)

# build the common parameter g
g = Element.random(pairing, G1)


def parameters():
    return pairing, g


class Registration_Server():
    def __init__(self):
        m = 2  # 多方安全计算需要的服务器数量
        n = 2  # 总共的RSU数量

        # build the public and private keys x
        self.x = Element.random(pairing, Zr)
        self.pkx = Element(pairing, G1, value=g ** self.x)

        # build the public and private keys y
        self.y = Element.random(pairing, Zr)
        self.pky = Element(pairing, G1, value=g ** self.y)

        # split keys for rsu
        self.rsu_id = [0x5fce79466c4a0d09, 0xb421105ff39ae3e6]
        self.rsu_yi = shamir_encryp(m, self.y, self.rsu_id, pairing)  # 切分密钥,获得rsu的私钥
        self.rsu_pki = dict()  # rsu的公钥
        for i, j in self.rsu_yi.items():
            self.rsu_pki[i] = Element(pairing, G1, value=g ** j)

        # initialize vehicle
        self.vehicle_id = []  # 车辆id
        self.vehicle_st = dict()  # 车辆注册状态

        # 预置车辆数量1w
        for i in range(10000000):
            id = getrandbits(64)
            self.vehicle_id.append(id)
            self.vehicle_st[id] = 1

    # 获取主公钥
    def get_pkx(self):
        return self.pkx

    # 获取验证公钥
    def get_pky(self):
        return self.pky

    # 获取rsu公钥
    def get_pki(self, rsu_id):
        return self.rsu_pki[rsu_id]

    # 设置车辆状态
    def set_vehicle_st(self, vehicle_id, st):
        if vehicle_id not in self.vehicle_id:
            return False
        self.vehicle_st[vehicle_id] = st

    # 车辆注册，获取vehicle私钥
    def register_vehicle(self, vehicle_id):
        if vehicle_id not in self.vehicle_id:
            self.vehicle_id.append(vehicle_id)
            self.set_vehicle_st(vehicle_id, 1)
        return Element(pairing, G1, value=hash_1(vehicle_id) ** self.x)

    # rsu注册,并返回私钥
    def register_rsu(self, i):
        id = self.rsu_id[i]
        return [id, self.rsu_yi[id], self.rsu_pki[id], self.pkx, self.pky, self.vehicle_st]

    # def get_token(self,exp):
    #     return scalar_mult(self.y,hash_1(exp))


# str to G
def hash_1(m):
    if type(m)!=bytes:
        if type(m) == int:
            m = m.to_bytes(8, byteorder='big')
        elif type(m) == str:
            m = m.encode("utf-8")
    return Element.from_hash(pairing, G1, m)


# str * G to G
def hash_2(m, G):
    if type(m) != bytes:
        if type(m) == int:
            m = m.to_bytes(8, byteorder='big')
        elif type(m) == str:
            m = m.encode("utf-8")
    G = str(G).encode("utf-8")
    return Element.from_hash(pairing, G1, m + G)


# Gt to G
def hash_3(G):
    m = G[0] + G[1]
    m = m.to_bytes(8, byteorder='big')
    return Element.from_hash(pairing, G1, m)


# G to base64
def hash_4(G):
    m = str(G).encode("utf-8")
    h = hashlib.sha256(m).digest()
    return base64.b64encode(h)
