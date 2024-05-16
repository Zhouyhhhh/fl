#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
import os
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, gtsrb_iid
from utils.options import args_parser
from models.Nets import MLP, CNNMnist, CNNCifar, CNNGtsrb
from models.Fed import FedAvg
from models.test import test_img
from models.MPC import M_agg
from time import time

from participant.Vehicle import Vehicle
from participant.RSU import RSU, get_exp
from participant.Res_Sever import Registration_Server

from beaver.beaver import beaver_check

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'gtsrb':
        trans_gtsrb = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor(),
             transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))])
        dataset_train = datasets.GTSRB('../data/gtsrb', split='train', download=True, transform=trans_gtsrb)
        dataset_test = datasets.GTSRB('../data/gtsrb', split='test', download=True, transform=trans_gtsrb)
        if args.iid:
            dict_users = gtsrb_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in gtsrb')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'gtsrb':
        net_glob = CNNGtsrb(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    # 注册服务器
    rs = Registration_Server()

    # rsu注册
    rsu1 = RSU()
    rsu2 = RSU()

    rsu1.register(rs.register_rsu(0))
    rsu2.register(rs.register_rsu(1))

    # 车辆注册，分配车辆训练的数据库，并得到车辆的私钥
    vehicle = []
    for i in range(args.num_users):
        v = Vehicle(args=args, dataset=dataset_train, idxs=dict_users[i])
        v.set_skc(rs.register_vehicle(v.get_id()))
        vehicle.append(v)
    print('{} has register successfully'.format(args.num_users))

    T_1 = []
    T_2 = []
    T_1_total = 0
    T_2_total = 0
    t_train = []

    AccTrain = []
    LossTrain = []
    AccTest = []
    LossTest = []

    fed_time = []

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        print('.................................................')
        print('{} epochs federated learning'.format(iter))
        fedtime1 = time()
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            print(idx)
            # 选择rsu
            rsu1_id, rus1_pk = rsu1.get_rsu()
            rsu2_id, rus2_pk = rsu2.get_rsu()
            rsu = [[rsu1_id, rus1_pk], [rsu2_id, rus2_pk]]
            if not vehicle[idx].get_is_token():
                req1, req2, beaver1, beaver2, t = vehicle[idx].request(net=copy.deepcopy(net_glob).to(args.device),
                                                                       rsu=rsu)
                t_train.append(t)

                judge = beaver_check(beaver1, beaver2)
                if judge==False:
                    print('wrong beaver')

                exp = get_exp(vehicle[idx].get_id())

                t1_total = time()
                sign1, t1, w1 = rsu1.authentication(req1, False, exp)
                sign2, t2, w2 = rsu2.authentication(req2, False, exp)
                t2_total = time()
                T_1_total += t2_total - t1_total
                print('total auth 1:', (t2_total - t1_total) / 2)
                print('train time:', t)
                T_1.append(t1)
                T_1.append(t2)

                sign = {**sign1, **sign2}
                token = vehicle[idx].set_exp_token(sign, exp)
            elif vehicle[idx].get_is_token():
                req1, req2, beaver1, beaver2, t = vehicle[idx].request(net=copy.deepcopy(net_glob).to(args.device),
                                                                       rsu=rsu)
                t_train.append(t)

                judge = beaver_check(beaver1, beaver2)
                if judge == False:
                    print('wrong beaver')

                t1_total = time()
                t1, w1 = rsu1.authentication(req1)
                t2, w2 = rsu2.authentication(req2)
                t2_total = time()
                T_2_total += t2_total - t1_total
                print('total auth 2:', (t2_total - t1_total) / 2)
                print('train time:', t)
                T_2.append(t1)
                T_2.append(t2)
            w = M_agg(w1, w2, w_glob)
            w_locals.append(copy.deepcopy(w))

            fedtime2 = time()

        # update global weights
        w_glob = FedAvg(w_locals)
        fedtime2 = time()
        fed_time.append((fedtime2 - fedtime1) / 10)
        # # copy weight to net_glob
        # t1 = time()
        net_glob.load_state_dict(w_glob)

        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)

        AccTrain.append(acc_train.item())
        AccTest.append(acc_test.item())
        LossTrain.append(loss_train)
        LossTest.append(loss_test)

        # t2 = time()
        # print('test:', t2 - t1)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Training loss: {:.2f}".format(loss_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        print("Testing loss: {:.2f}".format(loss_test))

    l1 = len(T_1)
    l2 = len(T_2)
    print('\n')
    print('.................................................')
    t = np.array(t_train)
    print('train:', np.mean(t))

    print('auth 1:', T_1_total / l1, l1)
    T1 = np.array(T_1)
    T2 = np.array(T_2)
    T1 = np.mean(T1, axis=0)
    T2 = np.mean(T2, axis=0)
    print('check id and st:', T1[0])
    print('generate exp and communicate: unknown')
    print('compute K:', T1[1])
    print('decrypt w and get bytes_w:', T1[2])
    print('generate str:', T1[3])
    print('authentication 1:', T1[4])
    print('generate sign:', T1[5])
    print('\n')

    print('auth 2:', T_2_total / l2, l2)
    print('check id and exp: unknown')
    print('compute K:', T2[0])
    print('decrypt w and get bytes_w:', T2[1])
    print('generate str:', T2[2])
    print('authentication 2:', T2[3])

    # w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
    # w = local.train(net=copy.deepcopy(net_glob).to(args.device))
    # if args.all_clients:
    #     w_locals[idx] = copy.deepcopy(w)
    # else:
    #     w_locals.append(copy.deepcopy(w))
    # loss_locals.append(copy.deepcopy(loss))

    # w_A, w_B = M_share(w)
    # w1 = M_agg(w_A, w_B)

    #     # update global weights
    #     w_glob = FedAvg(w_locals)
    #
    #     # copy weight to net_glob
    #     net_glob.load_state_dict(w_glob)
    #
    #     # print loss
    #     # loss_avg = sum(loss_locals) / len(loss_locals)
    #     # print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
    #     # loss_train.append(loss_avg)
    #
    # # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_train)), loss_train)
    # plt.ylabel('train_loss')
    # plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    #
    # testing
    net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    print("Testing loss: {:.2f}".format(loss_test))

    # chart 1
    matplotlib.rcParams['axes.unicode_minus'] = False
    label_list = ['gradient calculation', 'authentication 1', 'generate token']  # 横坐标刻度显示值
    num_list1 = [T1[1] + T1[2] + T1[3], T1[4], T1[5]]  # 纵坐标值1
    x = range(len(num_list1))
    # 绘制条形图
    rects1 = plt.bar(x, height=num_list1, width=0.6, alpha=0.7, color='blue')

    # 设置y轴属性
    plt.ylim(0, 0.035)
    plt.ylabel('time(ms)')

    # 设置x轴属性
    plt.xticks([index for index in x], label_list)
    plt.xlabel("authentication 1")

    # 显示文本
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha='center', va='bottom')

    plt.savefig("chart 1")

    plt.clf()

    # chart 2
    x = range(10)
    y = []
    for i in range(50):
        if (i + 1) % 5 == 0:
            y.append(fed_time[i])
    plt.plot(x, y)
    plt.ylim(0, 15)
    plt.xlabel('epochs')
    plt.ylabel('time(ms)')
    plt.savefig("chart 2")

    plt.clf()

    # chart 3
    print('len')
    print(len(AccTrain))
    print(AccTrain)

    x = range(10)
    xlable = [(i + 1) * 5 for i in range(10)]
    atrain = []
    atest = []
    ltrain = []
    ltest = []
    for i in range(50):
        if (i + 1) % 5 == 0:
            atrain.append(AccTrain[i])
            atest.append(AccTest[i])
            ltrain.append(LossTrain[i])
            ltest.append(LossTest[i])

    plt.plot(x, atrain, color='red', label='AccTrain', zorder=5)
    plt.plot(x, atest, color='blue', label='AccTest', zorder=10)
    plt.plot(x, ltrain, color='green', label='LossTrain', zorder=15)
    plt.plot(x, ltest, color='yellow', label='LossTest', zorder=20)
    plt.legend()

    # 设置y轴属性
    plt.ylim(0, 100)
    plt.ylabel('%')
    plt.xticks([index for index in x], xlable)
    plt.xlabel('epochs')

    plt.savefig("chart 3")

    plt.clf()
