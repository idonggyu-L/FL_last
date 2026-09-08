#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import random

from defense.RLRorigin import Rlr


from client.test import test_img
from defense.Fed import FedAvg
from defense.flame import flame
from defense.fltrust import fltrust
from defense.flare import flare
from defense.mkrum import multi_krum
from models.Nets import ResNet18, vgg19_bn
from client.Update import LocalUpdate
from utils.info import print_exp_details, write_info_to_accfile, get_base_info
from defense.multimetric import Multi_metrics
from defense.proto_bandit import ProtoBanditDefense, fedavg_noise
from utils.options import args_parser
from utils.sampling import cifar_iid, cifar_noniid, cifar_iid_fl, cifar_noniid_fl, dirichlet_partition


from client.Attacker import attacker
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import copy
import matplotlib
import pandas as pd

matplotlib.use('Agg')


def get_update(update, model):
    '''get the update weight'''
    update2 = {}
    for key, var in update.items():
        update2[key] = update[key] - model[key]
    return update2

def auxiliary_dataset(dataset, dataset_size):
    selected_class = 1
    central_dataset = set()
    class_idxs = [i for i, (_, label) in enumerate(dataset) if label == selected_class]
    selected_idxs = set(np.random.choice(class_idxs, dataset_size, replace=False))
    central_dataset.update(selected_idxs)
    return set(central_dataset)


def sample_round_users(num_users, round_size, malicious_clients, n_malicious):
    """Sample an exact malicious/benign split, with malicious clients first."""
    all_users = np.arange(int(num_users), dtype=np.int64)
    malicious_clients = np.asarray(malicious_clients, dtype=np.int64)
    n_malicious = int(n_malicious)
    round_size = int(round_size)
    if n_malicious == 0:
        return np.random.choice(all_users, round_size, replace=False)
    if n_malicious > len(malicious_clients):
        raise ValueError("round malicious count exceeds the malicious client pool")
    benign_clients = np.setdiff1d(all_users, malicious_clients, assume_unique=True)
    n_benign = round_size - n_malicious
    if n_benign < 0 or n_benign > len(benign_clients):
        raise ValueError("round benign count is incompatible with the client pools")
    selected_malicious = np.random.choice(
        malicious_clients, n_malicious, replace=False)
    selected_benign = np.random.choice(benign_clients, n_benign, replace=False)
    return np.concatenate((selected_malicious, selected_benign))


def probe_class_counts(total, n_classes, imbalance=1.0, seed=0):
    """합이 total이고 최소 1개인 balanced/long-tail 클래스별 probe 개수."""
    total = int(total)
    if total < n_classes:
        raise ValueError("pb_probe_n must be at least the number of classes")
    if imbalance < 1.0:
        raise ValueError("pb_probe_imbalance must be >= 1")
    if imbalance == 1.0:
        counts = np.full(n_classes, total // n_classes, dtype=np.int64)
        counts[:total % n_classes] += 1
        return counts
    weights = np.geomspace(float(imbalance), 1.0, n_classes)
    scaled = weights / weights.sum() * (total - n_classes)
    counts = np.floor(scaled).astype(np.int64) + 1
    remainder = total - int(counts.sum())
    order = np.argsort(-(scaled - np.floor(scaled)))
    counts[order[:remainder]] += 1
    rng = np.random.RandomState(int(seed))
    return counts[rng.permutation(n_classes)]



def write_file(filename, accu_list, back_list, args, analyse=False):
    write_info_to_accfile(filename, args)
    f = open(filename, "a")
    f.write("main_task_accuracy=")
    f.write(str(accu_list))
    f.write('\n')
    f.write("backdoor_accuracy=")
    f.write(str(back_list))
    if args.defence == "krum":
        krum_file = filename + "_krum_dis.csv"
        df = pd.DataFrame(args.krum_distance)
        df.to_csv(krum_file, index=False)
    elif args.defence == "mm":
        mm_file = filename + "_mm_dis1.csv"
        df = pd.DataFrame(args.mm1)
        # print(df)
        df.to_csv(mm_file, index=False)
        mm_file = filename + "_mm_dis2.csv"
        df = pd.DataFrame(args.mm2)
        df.to_csv(mm_file, index=False)
        mm_file = filename + "_mm_dis3.csv"
        df = pd.DataFrame(args.mm3)
        df.to_csv(mm_file, index=False)
        # df = pd.DataFrame(args.mm4)
        # torch.save(args.krum_distance, krum_file)
    if analyse == True:
        need_length = 10
        acc = accu_list[-need_length:]
        back = back_list[-need_length:]
        best_acc = round(max(acc), 2)
        average_back = round(np.mean(back), 2)
        best_back = round(max(back), 2)
        f.write('\n')
        f.write('BBSR:')
        f.write(str(best_back))
        f.write('\n')
        f.write('ABSR:')
        f.write(str(average_back))
        f.write('\n')
        f.write('max acc:')
        f.write(str(best_acc))
        f.write('\n')
        f.close()
        return best_acc, average_back, best_back
    f.close()




def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__ == '__main__':
    # parse args
    args = args_parser()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.attack == 'lp_attack':
        args.attack = 'adaptive'  # adaptively control the number of attacking layers
    args.device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    test_mkdir('./' + args.attack)
    print_exp_details(args)

    # load dataset and split users
    probe_dataset = None
    if args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # trans_cifar = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                                                  (0.2023, 0.1994, 0.2010))])
        dataset_train = datasets.CIFAR10(
            '../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10(
            '../data/cifar', train=False, download=True, transform=trans_cifar)
        probe_dataset = dataset_train
        if args.iid:
            if args.defence in ('fltrust', 'flare'):
                dict_users, central_dataset = cifar_iid_fl(dataset_train, args.num_users, 10, args.p, args=args)
            else:
                dict_users = cifar_iid_fl(dataset_train, args.num_users, 10, args.p, args=args)
        else:
            if args.defence in ('fltrust', 'flare'):
                dict_users, central_dataset = cifar_noniid_fl(dataset_train, args.num_users, 10, args.p,args=args)
            else:
                dict_users = cifar_noniid([x[1] for x in dataset_train], args.num_users, 10, args.p)

    elif args.dataset in ('mnist', 'fmnist'):
        # 흑백(1채널) → Grayscale(3)로 3채널 복제 → 기존 ResNet18(3채널) 그대로 사용 (A안)
        _tf = [transforms.Grayscale(num_output_channels=3)]
        if args.model == 'VGG':
            _tf.append(transforms.Resize(32))   # VGG19는 maxpool 5회 → 28x28이면 공간 소멸
        _tf += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        trans_gray3 = transforms.Compose(_tf)
        DS = datasets.FashionMNIST if args.dataset == 'fmnist' else datasets.MNIST
        _root = '../data/fmnist' if args.dataset == 'fmnist' else '../data/mnist'
        dataset_train = DS(_root, train=True, download=True, transform=trans_gray3)
        dataset_test = DS(_root, train=False, download=True, transform=trans_gray3)
        probe_dataset = dataset_train
        # cifar_noniid는 클래스별 균등개수를 가정 → MNIST/FMNIST는 클래스 불균등(5421~6742)이라
        # 각 클래스를 최소 클래스 크기로 잘라 균등화한 뒤 그 인덱스로만 분할 (CIFAR와 동일 파이프라인)
        _tgt = np.array(dataset_train.targets)
        _per = min((_tgt == c).sum() for c in range(10))
        _rng = np.random.RandomState(getattr(args, 'seed', 1))
        _bal = np.concatenate([_rng.permutation(np.where(_tgt == c)[0])[:_per] for c in range(10)])
        _bal.sort()
        _sub = Subset(dataset_train, _bal.tolist())
        _labels = [int(_tgt[i]) for i in _bal]
        if args.iid:
            _du = cifar_iid(_sub, args.num_users)
        else:
            _du = cifar_noniid(_labels, args.num_users, 10, args.p)   # 0..len(_bal)-1 로컬 인덱스
        # 로컬 인덱스 → 원본 dataset_train 인덱스로 복원
        dict_users = {k: set(int(_bal[i]) for i in v) for k, v in _du.items()}
        if args.defence in ('fltrust', 'flare'):
            # 서버 보유 root dataset (cifar 분기의 central_dataset_iid 와 동일 규칙: 클래스 균등)
            _per_c = max(1, int(args.server_dataset) // 10)
            _crng = np.random.RandomState(getattr(args, 'seed', 1) + 777)
            _cidx = []
            for c in range(10):
                _pool = np.where(_tgt == c)[0]
                _cidx.extend(_crng.choice(_pool, _per_c, replace=False).tolist())
            central_dataset = set(int(i) for i in _cidx)
            print('the central data', len(central_dataset))

    elif args.dataset == 'cifar100':
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                             std=[0.2675, 0.2565, 0.2761])])
        valid_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                                   std=[0.2675, 0.2565, 0.2761])])

        dataset_train = datasets.CIFAR100('../data/cifar100',
                                          train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR100('../data/cifar100',
                                         train=False, download=True, transform=valid_transform)
        probe_dataset = datasets.CIFAR100('../data/cifar100',
                                          train=True, download=True, transform=valid_transform)
        if args.iid:
            if args.defence in ('fltrust', 'flare'):
                dict_users, central_dataset = cifar_iid_fl(dataset_train, args.num_users, 100, args.p, args)
            else:
                dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            if args.defence in ('fltrust', 'flare'):
                dict_users_pre, central_dataset = cifar_noniid_fl(dataset_train, args.num_users, 100, args.p, args)
            else:
                dict_users_pre = cifar_noniid_fl(dataset_train, args.num_users, 100, args.p, args)
            dict_users = {}
            for key, val in dict_users_pre.items():
                dict_users[int(key/10)] = val


    # build model
    if args.model == 'VGG' and args.dataset == 'cifar':
        net_glob = vgg19_bn().to(args.device)
    elif args.model == 'VGG' and args.dataset == 'cifar100':
        net_glob = vgg19_bn(num_classes=100).to(args.device)
    elif args.model == 'VGG' and args.dataset in ('mnist', 'fmnist'):
        net_glob = vgg19_bn(num_classes=10).to(args.device)   # 입력은 32x32로 리사이즈
    elif args.model == "resnet" and args.dataset == 'cifar':
        net_glob = ResNet18().to(args.device)
        print(net_glob.state_dict().keys())
    elif args.model == "resnet" and args.dataset == 'cifar100':
        net_glob = ResNet18(num_classes=100).to(args.device)
    elif args.model == "resnet" and args.dataset in ('mnist', 'fmnist'):
        net_glob = ResNet18(num_classes=10).to(args.device)   # Grayscale(3)로 3채널화라 그대로
    else:
        exit('Error: unrecognized model')


    # 분할 방식: dirichlet 이면 위에서 만든 결정적 분할을 덮어쓴다(라벨 skew만, 수량 균등).
    if str(getattr(args, 'partition', 'group')) == 'dirichlet':
        _dnc = 100 if args.dataset == 'cifar100' else 10
        _dex = (central_dataset if (args.defence in ('fltrust', 'flare')
                                    and 'central_dataset' in dir() and central_dataset) else None)
        dict_users = dirichlet_partition(list(dataset_train.targets), args.num_users,
                                         _dnc, args.p, seed=getattr(args, 'seed', 1), exclude=_dex)
        if _dex:
            print('[main] Dirichlet: 서버 root %d장을 클라 풀에서 제외' % len(_dex))
        _lab = np.asarray(dataset_train.targets)
        _mx = [np.bincount(_lab[np.asarray(sorted(v))], minlength=_dnc).max() / len(v)
               for v in dict_users.values()]
        print('[main] Dirichlet 분할 q=%.2f: 클라당 %d장, 최대 클래스 비중 평균 %.3f (min %.3f / max %.3f)'
              % (args.p, len(next(iter(dict_users.values()))), np.mean(_mx), np.min(_mx), np.max(_mx)))

    # 클라별 데이터 양 이질성. 기본 분할은 전원 정확히 같은 표본 수라 SGD 스텝 수가 동일해지고,
    # 그 결과 정상 클라의 ‖Δw‖ 산포가 CV 0.001 수준으로 붕괴한다(파라미터 축 탐지가 세팅 인공물이 됨).
    # var>0 이면 각 클라가 자기 샤드의 (1-var, 1] 비율만 사용 → 스텝 수가 달라져 현실적 산포가 생긴다.
    _csv = float(getattr(args, 'client_size_var', 0.0))
    if _csv > 0:
        _crng = np.random.RandomState(int(getattr(args, 'seed', 1)) + 4242)
        for _k in list(dict_users.keys()):
            _idx = np.asarray(sorted(dict_users[_k]), dtype=np.int64)
            _n = max(20, int(len(_idx) * (1.0 - _csv * _crng.rand())))
            dict_users[_k] = set(int(i) for i in _crng.choice(_idx, min(_n, len(_idx)), replace=False))
        _sz = sorted(len(v) for v in dict_users.values())
        print('[main] client_size_var=%.2f → 표본 수 min %d / 중앙 %d / max %d'
              % (_csv, _sz[0], _sz[len(_sz)//2], _sz[-1]))

    # defense init
    multi_metric = Multi_metrics(10, 1, args)

    proto_defense = None
    fednoise_eval_idx = None
    use_wandb = False
    proto_probe_loader = None
    observe_only = bool(getattr(args, 'pb_observe', 0))
    if observe_only and args.defence != 'avg':
        raise ValueError("--pb_observe 1 is a passive FedAvg observer and requires --defence avg")
    if observe_only and not getattr(args, 'pb_no_anchor', 0):
        raise ValueError("--pb_observe 1 requires anchor-free mode (--pb_no_anchor 1)")
    if args.defence == 'protobandit' or observe_only:
        if getattr(args, 'pb_no_anchor', 0):
            # 실제 client pool에서 train-holdout probe 생성 + 모든 client에서 제외.
            # CIFAR100은 random train augmentation 대신 동일 sample의 deterministic view를 사용한다.
            _nc = 100 if args.dataset == 'cifar100' else 10
            _counts = probe_class_counts(
                getattr(args, 'pb_probe_n', 200), _nc,
                getattr(args, 'pb_probe_imbalance', 1.0),
                getattr(args, 'pb_probe_imbalance_seed', 0))
            _tg = np.array(dataset_train.targets)
            _available = set()
            for _v in dict_users.values():
                _available.update(int(i) for i in _v)
            _rng = np.random.RandomState(
                int(getattr(args, 'seed', 1)) + int(getattr(args, 'pb_probe_imbalance_seed', 0)))
            _pidx = []
            for _c, _need in enumerate(_counts):
                _ci = np.array(sorted(_available.intersection(np.where(_tg == _c)[0])), dtype=np.int64)
                _rng.shuffle(_ci)
                if len(_ci) < int(_need):
                    raise ValueError("class %d has %d available samples, needs %d for probe"
                                     % (_c, len(_ci), int(_need)))
                _pidx += _ci[:int(_need)].tolist()
            _pset = set(int(i) for i in _pidx)
            for _k in list(dict_users.keys()):
                dict_users[_k] = [i for i in dict_users[_k] if int(i) not in _pset]
            proto_probe_loader = DataLoader(Subset(probe_dataset, sorted(_pset)),
                                            batch_size=getattr(args, 'bs', 64), shuffle=False)
            print("[main] train-holdout probe: %d장, class counts=%s, 클라에서 제외"
                  % (len(_pset), _counts.tolist()))
        proto_defense = ProtoBanditDefense(net_glob, dataset_test, args, probe_loader=proto_probe_loader)
    # fednoise(노이즈만)·baseline(fltrust/flare/flame/avg)은 별도 셋업 없이 full test 로 평가

    # wandb 로깅 (모든 defence; project는 --wandb_project)
    if True:
        try:
            import wandb
            _rule_on = bool(getattr(args, 'pb_rule', 0))
            _observe_on = bool(getattr(args, 'pb_observe', 0))
            _na_tag = ("_noanchor_probe%d" % getattr(args, 'pb_probe_n', 200)
                       if getattr(args, 'pb_no_anchor', 0) else "")
            _method = "fedavg-observe" if _observe_on else ("rule" if _rule_on else args.defence)
            _imb_tag = (f"_imb{getattr(args,'pb_probe_imbalance',1.0)}"
                        f"s{getattr(args,'pb_probe_imbalance_seed',0)}")
            _rule_tag = (f"-{getattr(args,'pb_rule_dets','relmat,flip').replace(',','-')}"
                         f"_d{getattr(args,'pb_rule_drop',0.0)}k{getattr(args,'pb_rule_keep',0.5)}{_na_tag}"
                         if _rule_on else "")
            wandb.init(project=getattr(args, 'wandb_project', 'LGA-protobandit'),
                       name=(f"{args.dataset}_{args.model}_{args.attack}_{_method}{_rule_tag}{_imb_tag}"
                             f"_noise{getattr(args,'noise',0)}_mal{getattr(args,'malicious',0)}"
                             f"_q{args.p}_seed{args.seed}_ep{args.epochs}"),
                       config={'attack': args.attack, 'defence': args.defence,
                               'model': args.model, 'dataset': args.dataset,
                               'epochs': args.epochs, 'lr': args.lr,
                               'seed': args.seed, 'p': args.p, 'iid': args.iid,
                               'malicious': getattr(args, 'malicious', None),
                               'noise': getattr(args, 'noise', 0.0),
                               'pb_rule': int(_rule_on),
                               'pb_rule_dets': getattr(args, 'pb_rule_dets', None) if _rule_on else None,
                               'pb_rule_drop': getattr(args, 'pb_rule_drop', None) if _rule_on else None,
                               'pb_rule_keep': getattr(args, 'pb_rule_keep', None) if _rule_on else None,
                               'pb_no_anchor': int(getattr(args, 'pb_no_anchor', 0)),
                               'pb_probe_n': (getattr(args, 'pb_probe_n', None)
                                              if getattr(args, 'pb_no_anchor', 0) else None),
                               'pb_probe_imbalance': getattr(args, 'pb_probe_imbalance', None),
                               'pb_probe_imbalance_seed': getattr(args, 'pb_probe_imbalance_seed', None),
                               'pb_observe': int(_observe_on),
                               'pb_observe_every': getattr(args, 'pb_observe_every', None),
                               'bandit': ('none' if (_rule_on or args.defence != 'protobandit')
                                          else 'NeuralTS')})
            use_wandb = True
        except Exception as e:
            print("[wandb] init failed, logging disabled:", e)

    rlr = Rlr(args)
    args.local_bsr = []
    if args.log_distance == True:
        args.krum_distance = []
        args.krum_layer_distance = []
    args.mm1 = []
    args.mm2 = []
    args.mm3 = []

    # attack init
    if args.attackT == 0:
        args.attackT = 1
    prev_update = None
    args.attack_layers = []  # keep LSA
    if args.attack == "dba":
        args.dba_sign = 0  # control the index of group to attack


    # training
    net_glob.train()
    w_glob = net_glob.state_dict()
    loss_train = []

    base_info = get_base_info(args)
    filename = './' + args.attack + '/accuracy_file_{}.txt'.format(base_info)  # log hyperparameters

    val_acc_list = [0.0001]  # Acc list
    backdoor_acculist = [0]  # BSR list


    # 악성 풀. 기본(block)은 0..n-1 인데, 이 코드의 non-IID 분할은 dict_users[group*10+j] 라
    # 클라 번호가 데이터 분포를 결정한다 → block 이면 악성 전원이 같은 그룹(=같은 편중 클래스)이 되어
    # "공격 전부터 분포가 다르고 또래가 없는" 교란이 생긴다. spread 는 그룹마다 1명씩 뽑아
    # 악성/정상의 데이터 분포를 동일하게 맞춘다(탐지되면 오직 공격 때문).
    _n_mal = int(args.num_users * args.malicious)
    malicious_list = []
    if str(getattr(args, 'mal_pool', 'block')) == 'spread':
        # 클라 번호 공간에 균등 배치. cifar10(그룹당 10명)이면 그룹마다 정확히 1명 →
        # 악성/정상 분포가 동일해진다. cifar100(그룹당 1명)이면 편중 클래스가 흩어진다.
        _step = max(1, int(args.num_users) // max(_n_mal, 1))
        for i in range(_n_mal):
            malicious_list.append((i * _step) % int(args.num_users))
    else:
        for i in range(_n_mal):
            malicious_list.append(i)
    print('[main] malicious pool (%s): %s' % (getattr(args, 'mal_pool', 'block'), malicious_list))

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    for iter in range(args.epochs):
        args.iter = iter
        net_list = []
        loss_locals = []
        if not args.all_clients:
            w_locals = []
            w_updates = []
        m = max(int(args.frac * args.num_users), 1)  # number of clients in each round
        if iter >= args.attackT and (iter-args.attackT)%args.frequency==0 and iter <= args.end:  # start attack only when Acc overtakes backdoor_begin_acc
            attack_number = int(args.malicious * m)  # number of malicious clients in a single round
            print('attack_number')
        else:
            attack_number = 0
        n_mal_this_round = attack_number  # 진단용: w_locals 앞쪽 n개가 악성(주입 순서)
        idxs_users = sample_round_users(
            args.num_users, m, malicious_list, n_mal_this_round)
        args._round_client_ids = [int(i) for i in idxs_users]   # 덤프용: 클라 정체 기록

        mal_weight=[]
        mal_loss=[]

        if iter != 0:
            prev_prev_global_w = copy.deepcopy(prev_global_w)
            prev_global_w = copy.deepcopy(net_glob.state_dict())
            this_update = get_update(prev_global_w, prev_prev_global_w)
            prev_update = copy.deepcopy(this_update)


        for num_turn, idx in (enumerate(idxs_users)):
            if attack_number > 0 :  # upload models for malicious clients
                args.iter = iter
                m_idx = int(idx)
                mal_weight, loss, args.attack_layers = attacker(malicious_list, attack_number, args.attack, dataset_train, dataset_test, dict_users, net_glob, args, idx = m_idx,
                                                                prev_update=prev_update, dba_pos=num_turn)
                attack_number -= 1
                w = mal_weight[0]
            else:  # upload models for benign clients
                local = LocalUpdate(
                    args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(
                    net=copy.deepcopy(net_glob).to(args.device))

            w_updates.append(get_update(w, w_glob))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            new_net = copy.deepcopy(net_glob)
            new_net.load_state_dict(w)
            net_list.append(new_net)
            loss_locals.append(copy.deepcopy(loss))

        if iter == 0:
            prev_prev_global_w = copy.deepcopy(net_glob.state_dict())
            prev_global_w = copy.deepcopy(net_glob.state_dict())
            for net_id, net in enumerate(net_list):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        prev_global_w[key] = net_para[key] / len(net_list)
                else:
                    for key in net_para:
                        prev_global_w[key] += net_para[key] / len(net_list)

        if observe_only:
            _round = iter + 1
            _every = max(1, int(getattr(args, 'pb_observe_every', 50)))
            _last = max(0, int(getattr(args, 'pb_observe_last', 10)))
            if _round % _every == 0 or _round > args.epochs - _last:
                proto_defense.observe(w_locals, n_malicious=n_mal_this_round, round_idx=_round)
        # 마지막 N라운드 클라 weight(state_dict) 보존 — 임베딩/weight 분석용 (글로벌 아닌 각 클라)
        _wdir = getattr(args, 'pb_save_client_weights', '')
        if _wdir and (iter + 1) > args.epochs - int(getattr(args, 'pb_save_last', 10)):
            os.makedirs(_wdir, exist_ok=True)
            torch.save({'w_locals': w_locals, 'n_malicious': int(n_mal_this_round),
                        'round': iter + 1, 'idxs_users': list(idxs_users)},
                       os.path.join(_wdir, 'round%03d.pt' % (iter + 1)))

        if args.defence == 'avg':  # no defence
            w_glob = FedAvg(w_locals)
        elif args.defence == 'mkrum':
            selected_client = multi_krum(w_updates, args.k, args, multi_k=True)
            print(selected_client)
            w_glob = FedAvg([w_locals[x] for x in selected_client])
        elif args.defence == 'rlr':
            new_model = FedAvg(w_locals)
            w_glob = rlr.aggregate_updates(copy.deepcopy(net_glob), new_model, net_list)
        elif args.defence == 'fltrust':
            local = LocalUpdate(
                args=args, dataset=dataset_train, idxs=central_dataset)
            fltrust_norm, loss = local.train(
                net=copy.deepcopy(net_glob).to(args.device))
            fltrust_norm = get_update(fltrust_norm, w_glob)
            w_glob = fltrust(w_updates, fltrust_norm, w_glob, args)
        elif args.defence == 'flame':
            w_glob = flame(w_locals, w_updates, w_glob, args)
        elif args.defence == 'flare':
            # flare 내부는 (dataset, central 인덱스)로 PLR 추출 → central은 train 인덱스라 dataset_train 전달
            w_glob = flare(w_updates, w_locals, copy.deepcopy(net_glob),
                           central_dataset, dataset_train, w_glob, args)
        elif args.defence == 'mm':
            selected_client,os_dis, length_dis, manhattan_dis = multi_metric.exec_cuda(net_list)
            w_glob = FedAvg([w_locals[x] for x in selected_client])
        elif args.defence == 'protobandit':
            w_glob = proto_defense.aggregate(w_locals, w_glob, net_glob, n_malicious=n_mal_this_round)
        elif args.defence == 'fednoise':
            w_glob = fedavg_noise(w_locals, w_glob, getattr(args, 'noise', 0.001))
        else:
            print("Wrong Defense Method")
            os._exit(0)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # 매 라운드 방어 액션 wandb 기록 (protobandit)
        if use_wandb and proto_defense is not None and proto_defense.last_meta is not None:
            wandb.log(proto_defense.last_meta, step=iter)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        eval_now = (iter % 20 == 0 or iter > 280)
        if args.defence in ('protobandit', 'fednoise') or observe_only:
            # 5라운드마다 + 마지막 10라운드는 매 라운드(최종표 mean±std용)
            eval_now = (iter % 5 == 0 or iter >= args.epochs - 10)
        if eval_now:
            eval_set = dataset_test
            if args.defence == 'protobandit':
                # anchor-free(eval_idx=None)면 full test, 아니면 방어풀과 disjoint한 eval_idx
                eval_set = (dataset_test if proto_defense.eval_idx is None
                            else Subset(dataset_test, proto_defense.eval_idx))
            # fednoise/baseline(fltrust/flare/flame/avg)은 default eval_set=full test 사용
            acc_test, _, back_acc = test_img(
                    copy.deepcopy(net_glob), eval_set, args, test_backdoor=True)
            print("Main accuracy: {:.2f}".format(acc_test))
            print("Backdoor accuracy: {:.2f}".format(back_acc))
            if use_wandb:
                wandb.log({'main_accuracy': float(acc_test),
                           'backdoor_success_rate': float(back_acc)}, step=iter)
            val_acc_list.append(acc_test.item())
            backdoor_acculist.append(back_acc)
            write_file(filename, val_acc_list, backdoor_acculist, args)

    best_acc, absr, bbsr = write_file(filename, val_acc_list, backdoor_acculist, args, True)


    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))


