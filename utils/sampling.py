#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
from collections import defaultdict

import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_iid_fl(dataset, num_users, center):
    all_idxs = [i for i in range(len(dataset)) if i not in center]
    print('len of train data',len(all_idxs))
    num_items = int(len(all_idxs) / num_users)
    # print(num_items)
    dict_users = {}
    # dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_noniid(dataset_label, num_clients, num_classes, q):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    proportion = non_iid_distribution_group(dataset_label, num_clients, num_classes, q)
    dict_users = non_iid_distribution_client(proportion, num_clients, num_classes)
    return dict_users
def cifar_iid_fl(dataset, num_clients, num_classes, q, args=None):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    while True:
        try:
            if args.defence in ('fltrust', 'flare'):
                center_indices = set(central_dataset_iid(dataset, args.server_dataset, num_classes))
                all_idxs = [i for i in range(len(dataset)) if i not in center_indices]
            else:
                all_idxs = [i for i in range(len(dataset)) ]
            print('len of train data', len(all_idxs))
            num_items = int(len(all_idxs) / num_clients)
            # print(num_items)
            dict_users = {}
            # dict_users, all_idxs = {}, [i for i in range(len(dataset))]
            for i in range(num_clients):
                dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
                all_idxs = list(set(all_idxs) - dict_users[i])
            break
        except ValueError as e:
            # 这里可以添加一些打印信息，方便查看是哪里出现了问题
            print(f"{e},re-sampling data...")
            continue
    if args.defence in ('fltrust', 'flare'):
        print('the central data', len(center_indices))
        return dict_users, center_indices
    return dict_users


def cifar_noniid_fl(dataset, num_clients, num_classes, q, args=None):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    while True:
        try:
            if args.defence in ('fltrust', 'flare'):
                center_indices = set(central_dataset_iid(dataset, args.server_dataset, num_classes))
                dataset_label = [x[1] for i, x in enumerate(dataset) if i not in center_indices]
            else:
                dataset_label = [x[1] for i, x in enumerate(dataset)]
            proportion = non_iid_distribution_group(dataset_label, num_clients, num_classes, q)
            dict_users = non_iid_distribution_client(proportion, num_clients, num_classes)
            break
        except ValueError as e:
            print(f"{e},re-sampling data...")
            continue
    if args.defence in ('fltrust', 'flare'):
        print('the central data', len(center_indices))
        return dict_users, center_indices
    return dict_users

def cifar100_iid_fl(dataset, num_clients, num_classes, q, args=None):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """

    # proportion = non_iid_distribution_group(dataset_label, num_clients, num_classes, q)
    # dict_users = non_iid_distribution_client(proportion, num_clients, num_classes)
    while True:
        try:
            if args.defence == 'fltrust':
                center_indices = set(central_dataset(dataset, args.server_dataset, num_classes))
                all_idxs = [i for i in range(len(dataset)) if i not in center_indices]
            else:
                all_idxs = [i for i in range(len(dataset)) ]
            print('len of train data', len(all_idxs))
            num_items = int(len(all_idxs) / num_clients)
            # print(num_items)
            dict_users = {}
            # dict_users, all_idxs = {}, [i for i in range(len(dataset))]
            for i in range(num_clients):
                dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
                all_idxs = list(set(all_idxs) - dict_users[i])
            break
        except ValueError as e:
            # 这里可以添加一些打印信息，方便查看是哪里出现了问题
            print(f"出现错误: {e}，正在重新分配数据...")
            continue
    if args.defence == 'fltrust':
        print('the central data', len(center_indices))
        return dict_users, center_indices
    return dict_users

def non_iid_distribution_group(dataset_label, num_clients, num_classes, q):
    dict_users, all_idxs = {}, [i for i in range(len(dataset_label))]
    for i in range(num_classes):
        dict_users[i] = set([])
    for k in range(num_classes):
        idx_k = np.where(np.array(dataset_label) == k)[0]
        num_idx_k = len(idx_k)
        
        selected_q_data = set(np.random.choice(idx_k, int(num_idx_k*q) , replace=False))
        dict_users[k] = dict_users[k]|selected_q_data
        idx_k = list(set(idx_k) - selected_q_data)
        all_idxs = list(set(all_idxs) - selected_q_data)
        for other_group in range(num_classes):
            if other_group == k:
                continue
            selected_not_q_data = set(np.random.choice(idx_k, int(num_idx_k*(1-q)/(num_classes-1)) , replace=False))
            dict_users[other_group] = dict_users[other_group]|selected_not_q_data
            idx_k = list(set(idx_k) - selected_not_q_data)
            all_idxs = list(set(all_idxs) - selected_not_q_data)
    print(len(all_idxs),' samples are remained')
    print('random put those samples into groups')
    num_rem_each_group = len(all_idxs) // num_classes
    for i in range(num_classes):
        selected_rem_data = set(np.random.choice(all_idxs, num_rem_each_group, replace=False))
        dict_users[i] = dict_users[i]|selected_rem_data
        all_idxs = list(set(all_idxs) - selected_rem_data)
    print(len(all_idxs),' samples are remained after relocating')
    return dict_users


def non_iid_distribution_client(group_proportion, num_clients, num_classes):
    num_each_group = num_clients // num_classes
    num_data_each_client = len(group_proportion[0]) // num_each_group
    dict_users, all_idxs = {}, [i for i in range(num_data_each_client*num_clients)]
    for i in range(num_classes):
        group_data = list(group_proportion[i])
        for j in range(num_each_group):
            selected_data = set(np.random.choice(group_data, num_data_each_client, replace=False))
            dict_users[i*10+j] = selected_data
            group_data = list(set(group_data) - selected_data)
            all_idxs = list(set(all_idxs) - selected_data)
    print(len(all_idxs),' samples are remained')
    return dict_users


def check_data_each_client(dataset_label, client_data_proportion, num_client, num_classes):
    for client in client_data_proportion.keys():
        client_data = dataset_label[list(client_data_proportion[client])]
        print('client', client, 'distribution information:')
        for i in range(num_classes):
            print('class ', i, ':', len(client_data[client_data==i])/len(client_data))


def central_dataset_iid(dataset, dataset_size, class_num):
    dataset_label = [x[1] for i, x in enumerate(dataset)]
    central_dataset = []
    if class_num > 0:
        for k in range(class_num):
            idx_k = np.where(np.array(dataset_label) == k)[0]
            central_dataset.extend(np.random.choice(
        idx_k, int(dataset_size/class_num), replace=False))
        print(type(central_dataset))
        return set(np.array(central_dataset))
    all_idxs = [i for i in range(len(dataset))]

    central_dataset = set(np.random.choice(
        all_idxs, dataset_size, replace=False))
    return central_dataset

def auxiliary_dataset(dataset, dataset_size):
    selected_class = 1
    central_dataset = set()
    class_idxs = [i for i, (_, label) in enumerate(dataset) if label == selected_class]
    selected_idxs = set(np.random.choice(class_idxs, dataset_size, replace=False))
    central_dataset.update(selected_idxs)
    return set(central_dataset)

def noniid_cifar100(dataset, num_clients, num_classes, q,args):

    min_size = 0
    K = num_classes
    N = len(dataset)
    net_dataidx_map = {}
    if args.defence == 'fltrust':
        center_indices = set(central_dataset_iid(dataset, args.server_dataset, num_classes))
        dataset_label = [x[1] for i, x in enumerate(dataset) if i not in center_indices]
    elif args.defence == 'flare':
        center_indices = set(auxiliary_dataset(dataset, 10))
        dataset_label = [x[1] for i, x in enumerate(dataset) if i not in center_indices]
    else:

        dataset_label = [x[1] for i, x in enumerate(dataset)]

    while (min_size < 10):
        print(min_size,'min size')
        idx_batch = [[] for _ in range(num_clients)]
        # for each class in the dataset
        for k in range(K):
            idx_k = np.where(np.array(dataset_label) == k)[0]
            # print(len(idx_k))
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(q, num_clients))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_clients):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    if args.defence == 'fltrust' or args.defence == 'flare':
        print(f'central dataset {len(center_indices)}')
        return net_dataidx_map, center_indices

    return net_dataidx_map

def central_dataset(dataset, dataset_size, class_num):
    # dataset_label = [x[1] for i, x in enumerate(dataset)]
    central_dataset = []
    # if class_num > 0:
    #     for k in range(class_num):
    #         idx_k = np.where(np.array(dataset_label) == k)[0]
    #         central_dataset.extend(np.random.choice(
    #             idx_k, int(dataset_size / class_num), replace=False))
    #     print(type(central_dataset))
    #     return set(np.array(central_dataset))
    print(len(dataset))
    all_idxs = [i for i in range(len(dataset))]
    print(all_idxs)
    central_dataset = set(np.random.choice(
        all_idxs, dataset_size, replace=False))
    return central_dataset

def get_noniid(dataset, num_clients, num_classes, q, args):
    cifar_classes = {}
    if args.defence == 'fltrust':
        center_indices = set(central_dataset(dataset, args.server_dataset, num_classes))
        # print(center_indices)
    elif args.defence == 'flare':
        center_indices = set(auxiliary_dataset(dataset, 10))
    else:
        center_indices = set()
        # dataset_label = [x[1] for i, x in enumerate(dataset) if i not in center_indices]
    for ind, x in enumerate(dataset):
        _, label = x
        if (args.defence == 'fltrust' or args.defence == 'flare') and ind in center_indices:
            continue
        if label in cifar_classes:
            cifar_classes[label].append(ind)
        else:
            cifar_classes[label] = [ind]
    class_size = len(cifar_classes[0])
    per_participant_list = defaultdict(list)
    no_classes = len(cifar_classes.keys())

    for n in range(no_classes):
        random.shuffle(cifar_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(
            np.array(num_clients * [q]))
        for user in range(num_clients):
            no_imgs = int(round(sampled_probabilities[user]))
            sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
            per_participant_list[user].extend(sampled_list)
            cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
    if args.defence == 'fltrust' or args.defence == 'flare':
        print(f'central dataset {len(center_indices)}')
        return per_participant_list, center_indices
    return per_participant_list

if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)


def dirichlet_partition(dataset_label, num_clients, num_classes, q, seed=1, exclude=None):
    """클라마다 p_i ~ Dir(q·1_K) 로 클래스 비율을 뽑고 그 비율대로 표본 배정.
    표본 수는 클라마다 동일(N/num_clients) → 수량은 균등, 라벨만 치우침.
    q 가 작을수록 특정 클래스에 몰림(표준 Dirichlet 농도 관례)."""
    rng = np.random.RandomState(int(seed))
    labels = np.asarray(dataset_label)
    _ex = set() if exclude is None else set(int(i) for i in exclude)   # fltrust/flare 서버 root 제외
    pool = [rng.permutation([i for i in np.where(labels == c)[0] if int(i) not in _ex]).tolist()
            for c in range(num_classes)]
    n_per = (len(labels) - len(_ex)) // num_clients
    dict_users = {}
    for i in range(num_clients):
        p = rng.dirichlet([float(q)] * num_classes)
        cnt = rng.multinomial(n_per, p)
        picked = []
        for c in range(num_classes):
            take = int(min(cnt[c], len(pool[c])))
            if take:
                picked += pool[c][:take]
                pool[c] = pool[c][take:]
        short = n_per - len(picked)                       # 소진된 클래스 몫은 남은 풀에서 보충
        avail = [c for c in range(num_classes) if pool[c]]
        while short > 0 and avail:
            c = avail[rng.randint(len(avail))]
            picked.append(pool[c].pop())
            if not pool[c]:
                avail.remove(c)
            short -= 1
        dict_users[i] = set(int(x) for x in picked)
    return dict_users
