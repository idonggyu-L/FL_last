#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # save file 
    parser.add_argument('--save', type=str, default='save',
                        help="dic to save results (ending without /)")
    parser.add_argument('--init', type=str, default='None',
                        help="location of init model")
    # federated arguments
    parser.add_argument('--epochs', type=int, default=300,
                        help="rounds of training")
    parser.add_argument('--num_users', type=int,
                        default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help="the fraction of clients: C")
    parser.add_argument('--malicious',type=float,default=0.1, help="proportion of mailicious clients")
    
    #***** badnet labelflip layerattack updateflip get_weight  adaptive****
    parser.add_argument('--attack', type=str,
                        default='ours', help='attack method')
    parser.add_argument('--ada_mode', type=int,
                        default=1, help='adaptive attack mode')
    parser.add_argument('--ada_assume', type=str, default='',
                        choices=['', 'krum', 'multikrum', 'flame'],
                        help="adaptive 공격자가 상정하는 방어(실제 방어와 분리). ''=기존 동작")
    parser.add_argument('--poison_frac', type=float, default=1,
                        help="fraction of dataset to corrupt for backdoor attack, 1.0 for layer attack")

    # *****local_ep = 3, local_bs=50, lr=0.1*******
    parser.add_argument('--local_ep', type=int, default=2,
                        help="the number of local epochs: E")
    parser.add_argument('--local_ep_mal', type=int, default=6)
    parser.add_argument('--local_bs', type=int, default=64,
                        help="local batch size: B")

    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate")
    parser.add_argument('--lr_mal', type=float, default=0.1,
                        help="learning rate")
    # model arguments
    #*************************model******************************#
    # resnet cnn VGG mlp Mnist_2NN Mnist_CNN resnet20 rlr_mnist
    parser.add_argument('--model', type=str,
                        default='resnet', help='model name')

    # other arguments
    #*************************dataset*******************************#
    # fashion_mnist mnist cifar
    parser.add_argument('--dataset', type=str,
                        default='mnist', help="name of dataset")
    
    
    
    #****0-avg, 1-fltrust 2-tr-mean 3-median 4-krum 5-muli_krum 6-RLR fltrust_bn fltrust_bn_lr****#
    parser.add_argument('--defence', type=str,
                        default='flame', help="strategy of defence")
    parser.add_argument('--k', type=int,
                        default=2, help="parameter of krum")
    # parser.add_argument('--iid', action='store_true',
    #                     help='whether i.i.d or not')
    parser.add_argument('--iid', type=int, default=0,
                        help='whether i.i.d or not')

 #************************atttack_label********************************#
    parser.add_argument('--partition', type=str, default='group', choices=['group','dirichlet'],
                        help="클라 분할: group=결정적 지배클래스 q(기존) / dirichlet=p_i~Dir(q), q 작을수록 강한 skew")
    parser.add_argument('--client_size_var', type=float, default=0.0,
                        help='클라별 표본 수 이질성. 각 클라가 샤드의 (1-var,1] 비율만 사용. 0=균등(기존)')
    parser.add_argument('--mal_pool', type=str, default='block', choices=['block','spread'],
                        help="악성 클라 선정: block=0..n-1(기본, 전원 같은 데이터 그룹) / spread=그룹마다 1명(분포 교란 제거)")
    parser.add_argument('--attack_label', type=int, default=5,
                        help="trigger for which label")
    
    parser.add_argument('--single', type=int, default=0,
                        help="single shot or repeated")
    # attack_goal=-1 is all to one
    parser.add_argument('--attack_goal', type=int, default=-1,
                        help="trigger to which label")
    # --attack_begin 70 means accuracy is up to 70 then attack
    parser.add_argument('--attack_begin', type=int, default=0,
                        help="the accuracy begin to attack")
    # search times
    parser.add_argument('--search_times', type=int, default=20,
                        help="binary search times")
    
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU ID, -1 for CPU")
    parser.add_argument('--robustLR_threshold', type=int, default=4, 
                        help="break ties when votes sum to 0")
    
    parser.add_argument('--server_dataset', type=int,default=200,help="number of dataset in server")
    parser.add_argument('--dataset_flare', type=int,default=10)
    parser.add_argument('--server_lr', type=float,default=1,help="number of dataset in server using in fltrust")
    
    
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="SGD momentum (default: 0.5)")
    
    
    parser.add_argument('--split', type=str, default='user',
                        help="train-test split type, user or sample")   
    #*********trigger info*********
    #  square  apple  watermark  
    parser.add_argument('--trigger', type=str, default='square',
                        help="Kind of trigger")
    parser.add_argument('--apple', type=int, default=None)
    parser.add_argument('--blend',  default=None)
    parser.add_argument('--watermark', type=int, default=None)
    parser.add_argument('--sig', default=None)
    parser.add_argument('--LFT', default=None)
    parser.add_argument('--local_ep_trigger',default=10,type=int)
    parser.add_argument('--atk_eps',default=0.1,type=float)
    parser.add_argument('--atk_eps_test',default=0.1,type=float)
    # mnist 28*28  cifar10 32*32
    parser.add_argument('--triggerX', type=int, default='27',
                        help="position of trigger x-aix") 
    parser.add_argument('--triggerY', type=int, default='27',
                        help="position of trigger y-aix")
    parser.add_argument('--dba_size', type=int, default=2,
                        help="DBA 4분할 각 조각 크기 s (전체 폭=2s+gap). 키우면 공격 강해짐")
    parser.add_argument('--dba_gap', type=int, default=0,
                        help="DBA 조각 사이 간격 g. >0이면 조각이 떨어진 독립 패턴(정통 DBA), 0이면 통짜 블록")

    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--wrong_mal', type=int, default=0)
    parser.add_argument('--right_ben', type=int, default=0)
    
    parser.add_argument('--mal_score', type=float, default=0)
    parser.add_argument('--ben_score', type=float, default=0)
    
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--noise', type=float, default=0.001)
    # protobandit 설정 (병렬 비교용)
    parser.add_argument('--pb_actions', type=str, default='flip,l2_to_center',
                        help="ACTION detector 목록(쉼표). 예: flip,relmat")
    parser.add_argument('--pb_reward', type=str, default='full', choices=['full', 'acc'],
                        help="full=acc+relmat-앵커+drift, acc=accuracy만")
    parser.add_argument('--pb_rule', type=int, default=0,
                        help="1=밴딧 대신 룰베이스(flip+relmat 고정선택)")
    parser.add_argument('--pb_rule_dets', type=str, default='relmat,flip',
                        help="룰베이스 결합 detector(쉼표). 예: relmat / flip / rotation / relmat,flip,rotation")
    parser.add_argument('--pb_no_anchor', type=int, default=1,
                        help="1=앵커 없이 또래비교만(anchor-free). flip/rot baseline은 z-score에서 상쇄되므로 불필요")
    parser.add_argument('--pb_probe_n', type=int, default=200,
                        help="anchor-free 시 프로토타입 probe 크기(클래스 균형, train-holdout)")
    parser.add_argument('--pb_probe_imbalance', type=float, default=1.0,
                        help="anchor-free probe 클래스 최대/최소 표본수 비율(1=균형, >1=long-tail)")
    parser.add_argument('--pb_probe_imbalance_seed', type=int, default=0,
                        help="불균형 probe의 클래스별 표본수 배치를 섞는 seed")
    parser.add_argument('--pb_dump_relmat', type=str, default='',
                        help="지정 시 매 라운드 전체 클라 관계행렬+악성마스크를 이 npz에 누적 저장(시각화용)")
    parser.add_argument('--pb_combine', type=str, default='sum', choices=['sum','inter'],
                        help="detector 결합: sum=z합산 후 밴드 / inter=detector별 밴드의 교집합(상쇄 회피)")
    parser.add_argument('--pb_noise_mode', type=str, default='norm', choices=['norm','flame'],
                        help="노이즈 스케일: norm=λ·median‖Δw‖/√P(기존) / flame=λ·S 원소별(FLAME 원식)")
    parser.add_argument('--pb_clip_q', type=float, default=0.0,
                        help='>0이면 선택된 업데이트를 clip_q×median_norm 으로 클리핑 후 집계(노이즈와 결합). 0=클리핑 없음')
    parser.add_argument('--pb_dump_light', type=int, default=0,
                        help='1이면 덤프 시 프로토타입만 저장하고 perturbation(11 forward)·accuracy 계산 생략')
    parser.add_argument('--pb_dump_protos', type=int, default=0,
                        help='1이면 덤프에 원본 프로토타입(K*D)도 저장 → cos/유클리드 등 임의 관계행렬 오프라인 재계산')
    parser.add_argument('--pb_dump_every', type=int, default=1,
                        help="관계행렬 dump 저장 주기(라운드 수; 메모리에는 매 라운드 누적)")
    parser.add_argument('--pb_observe', type=int, default=0,
                        help="1이면 defence=avg에서 로컬 통계만 수집하고 aggregation에는 관여하지 않음")
    parser.add_argument('--pb_observe_every', type=int, default=50,
                        help="passive observer의 임베딩/로컬-정확도 측정 간격")
    parser.add_argument('--pb_observe_last', type=int, default=10,
                        help="passive observer가 마지막에 매 라운드 측정할 라운드 수")
    parser.add_argument('--pb_save_client_weights', type=str, default='',
                        help="지정 시 마지막 N라운드의 각 클라 state_dict를 이 폴더에 round%%03d.pt로 저장(임베딩/weight 분석용)")
    parser.add_argument('--pb_save_last', type=int, default=10,
                        help="pb_save_client_weights로 저장할 마지막 라운드 수")
    parser.add_argument('--wandb_project', type=str, default='LGA-protobandit',
                        help="wandb project 이름")
    parser.add_argument('--pb_aug_abs', type=int, default=0,
                        help="1=aug(flip/rot) detector를 median 중심 |편차|로(양방향). 부호 backbone-의존 대응")
    parser.add_argument('--pb_warmup_rounds', type=int, default=0,
                        help="초반 N라운드는 pb_warmup_keep으로 더 깊게 컷(탐지 lock-on 유도). 0=off")
    parser.add_argument('--pb_warmup_keep', type=float, default=0.3,
                        help="warmup 구간의 keep 비율")
    parser.add_argument('--pb_perturb_peer', type=int, default=0,
                        help="1=flip/rot detector를 '자기차이(clean-perturbed)' 대신 'perturbed 행렬 또래비교'로. 리뷰어 반박 ablation")
    parser.add_argument('--pb_aug_tails', type=int, default=0,
                        help="1=aug detector를 양쪽 꼬리 분리(pos=max(z,0), neg=max(-z,0) 둘 다 combined에)")
    parser.add_argument('--pb_aug_weight', type=str, default='off', choices=['off','gap','max'],
                        help="detector 자동가중: gap=(top1-median)|z|, max=max|z|, off=균등. 무신호 detector 희석 방지")
    parser.add_argument('--pb_topk', type=int, default=0,
                        help="Top-k 자동선택: detector별 gap의 EMA 상위 k개만 combined 사용(무신호 배제). 0=off")
    parser.add_argument('--pb_rule_drop', type=float, default=0.0,
                        help="룰베이스 band 안쪽(최상위 신뢰) 버릴 비율 (도넛이면 >0)")
    parser.add_argument('--pb_rule_keep', type=float, default=0.5,
                        help="룰베이스 band 바깥 한계선 비율")
    parser.add_argument('--pb_evals', type=int, default=10,
                        help="라운드당 평가할 액션 수")
    parser.add_argument('--all_clients', action='store_true',
                        help='aggregation over all clients') 
    parser.add_argument('--tau', type=float, default=0.95,
                        help="threshold of LPA_ER")
    parser.add_argument('--debug', type=int, default=0, help="log debug info or not")
    parser.add_argument('--local_dataset', type=int, default=1, help="use local dataset for layer identification")
    parser.add_argument('--debug_fld', type=int, default=0, help="#1 save, #2 load")
    parser.add_argument('--decrease', type=float, default=0.3, help="proportion of dropped layers in robust experiments (used in mode11)")
    parser.add_argument('--increase', type=float, default=0.3, help="proportion of added layers in robust experiments (used in mode12)")
    parser.add_argument('--mode10_tau', type=float, default=0.95, help="threshold of mode 10")
    parser.add_argument('--cnn_scale', type=float, default=0.5, help="scale of cnn")
    parser.add_argument('--cifar_scale', type=float, default=1.0, help="scale of larger model")
    
    parser.add_argument('--num_layer', type=int, default=3, help="fixed number of layer attacks")
    
    parser.add_argument('--num_identification', type=int, default=1, help="fixed number of round to identify")
    parser.add_argument('--beta', type=float, default=0.5, help="weight of regularization loss in distance awareness attacks")
    parser.add_argument('--log_distance', type=bool, default=False, help="output krum distance")
    parser.add_argument('--scaling_attack_round', type=int, default=1, help="rounds of attack implements")
    parser.add_argument('--scaling_param', type=float, default=5, help="scaling up how many times")
    parser.add_argument('--p', type=float, default=0.5, help="level of non-iid")
    parser.add_argument('--frequency',type=int,default=1)
    parser.add_argument('--attackT',type=int,default=0)
    parser.add_argument('--end',type=int,default=100000)

    parser.add_argument('--alignUpdate',type=str,default='onlyg')
    parser.add_argument('--M',type=float, default=0)
    parser.add_argument('--iter',type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--scale_int',type=str,default='epoch')
    parser.add_argument('--BN',type=int,default=1)
    parser.add_argument('--integrate',default=0,type=int)
    parser.add_argument('--allacc', default=0,type=int)
    parser.add_argument('--neu',default=0,type=int)
    parser.add_argument('--ifour', default=0, type=int)
    parser.add_argument('--pool',default=0,type=int)
    parser.add_argument('--prev',default=1,type=int)
    parser.add_argument('--prevs',default=[],type=list)
    parser.add_argument('--attackIter', type=int, default=0)
    parser.add_argument('--krum_distance', type=list,default=[])
    parser.add_argument('--mm1', type=list,default=[])

    parser.add_argument('--mm2', type=list,default=[])
    parser.add_argument('--mm3', type=list,default=[])
    parser.add_argument('--A',type=float,default=0.5)
    parser.add_argument('--minA',type=float,default=0.1)
    parser.add_argument('--maxA',type=float,default=1)
    parser.add_argument('--midA',type=float,default=0.5)
    parser.add_argument('--pruning', type=int,default=0)
    parser.add_argument('--topp', type=float,default=0.03)
    # parser.add_argument('--',type=float,default=0.5)


    parser.add_argument('--dense_ratio', type=float, default=0.25)
    parser.add_argument('--se_threshold', type=float, default=1e-4,
                        help="num of workers for multithreading")
    parser.add_argument('--theta', type=float, default=0.5,
                        help="break ties when votes sum to 0")
    parser.add_argument('--anneal_factor', type=float, default=0.0001,
                        help="num of workers for multithreading")
    parser.add_argument('--mask_init', type=str, default="ERK")

    parser.add_argument('--dis_check_gradient', action='store_true', default=False)

    args = parser.parse_args()
    return args
