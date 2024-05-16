#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from time import time
from random import randrange, getrandbits
from struct import pack

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.MPC import M_share
from models.func_w import encrypt, decrypt
from shamir.shamir import shamir_decrypt
from beaver.beaver import beaver_generate

from participant.Res_Sever import parameters, hash_1, hash_2, hash_4
from pypbc import *

pairing, g = parameters()


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class Vehicle(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        self.skc = 0
        self.id = getrandbits(64)
        self.exp = 0
        self.token = 0
        self.K = dict()
        self.is_token = False

    def get_id(self):
        return self.id

    def set_skc(self, skc):
        self.skc = skc

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        # return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
        return net.state_dict()

    def request_without_token(self, w, rsu_id, rsu_pk):

        r1 = Element.random(pairing, Zr)
        R1 = hash_1(self.id) ** r1
        R2 = g ** r1

        r2 = Element.random(pairing, Zr)
        R3 = g ** r2

        K = hash_4(rsu_pk ** r2)
        if rsu_id not in self.K.keys():
            self.K[rsu_id] = K

        # 给梯度加密，并得到明文字节用于签名
        c, bytew = encrypt(w, K)

        TS = time()

        # ID+TS+m
        m = self.id.to_bytes(8, byteorder='big') + pack('d', TS) + bytew
        I = (self.skc + hash_2(m, R1)) ** r1

        return (self.id, TS, R1, R2, R3, I, c)

    def request_with_token(self, w, rsu_id, rsu_pk):
        r3 = Element.random(pairing, Zr)
        R3 = hash_1(self.exp) ** r3
        R4 = g ** r3

        K = hash_4(rsu_pk ** r3)
        if rsu_id not in self.K.keys():
            self.K[rsu_id] = K

        TS = time()

        # 给梯度加密，并得到明文字节用于签名
        c, bytew = encrypt(w, K)

        # ID+TS+m
        m = self.id.to_bytes(8, byteorder='big') + pack('d', TS) + bytew
        I = (hash_2(m, rsu_pk ** r3) + self.token) ** r3

        return (self.id, self.exp, I, R3, R4, TS, c)

    def request(self, net, rsu):
        t1 = time()
        w = self.train(net)
        t2 = time()
        w_A, w_B, n = M_share(w)
        beaver1, beaver2 = beaver_generate(n)

        if not self.is_token:
            req1 = self.request_without_token(w_A, rsu[0][0], rsu[0][1])
            req2 = self.request_without_token(w_B, rsu[1][0], rsu[1][1])
        elif self.is_token:
            req1 = self.request_with_token(w_A, rsu[0][0], rsu[0][1])
            req2 = self.request_with_token(w_B, rsu[1][0], rsu[1][1])
        return req1, req2, beaver1, beaver2, t2 - t1

    def set_exp_token(self, sign, exp):
        self.exp = exp
        self.token = shamir_decrypt(sign, g, pairing)
        self.is_token = True

    def get_is_token(self):
        return self.is_token
