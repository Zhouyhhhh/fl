import copy
import torch
import random
import numpy as np


def M_share(w):
    w_A = []
    w_B = []
    n=0
    for k in w.keys():
        shape = w[k].shape
        if len(shape) == 0:
            nump = 0
            w_A.append(nump)
            w_B.append(nump)
        elif len(w[k].shape) == 1:
            nump = np.array(w[k])
            rand = np.random.rand(shape[0])
            w_A.append(rand)
            w_B.append(nump - rand)
            n+=shape[0]
        elif len(w[k].shape) == 2:
            nump = np.array(w[k])
            rand = np.random.rand(shape[0], shape[1])
            w_A.append(rand)
            w_B.append(nump - rand)
            n += shape[0]*shape[1]
        elif len(w[k].shape) == 3:
            nump = np.array(w[k])
            rand = np.random.rand(shape[0], shape[1], shape[2])
            w_A.append(rand)
            w_B.append(nump - rand)
            n += shape[0] * shape[1]*shape[2]
        elif len(w[k].shape) == 4:
            nump = np.array(w[k])
            rand = np.random.rand(shape[0], shape[1], shape[2], shape[3])
            w_A.append(rand)
            w_B.append(nump - rand)
            n += shape[0] * shape[1] * shape[2]*shape[3]

    return w_A, w_B,n


def M_agg(w_a, w_b, dic):
    w = copy.deepcopy(dic)
    i = 0
    for k in w.keys():
        if type(w_a[i])==int and w_a[i] == 0:
            w[k] = torch.tensor(0)
        else:
            add = w_a[i] + w_b[i]
            tensor = torch.from_numpy(add)
            w[k] = tensor
        i += 1
    return w
