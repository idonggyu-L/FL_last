import time
import math

def print_exp_details(args):
    info = information(args)
    for i in info:
        print(i)
    write_info(args, info)
    
def write_info_to_accfile(filename, args):
    info = information(args)
    f = open(filename, "w")
    for i in info:
        f.write(i)
        f.write('\n')
    f.close()    
    
def write_info(args, info):
    f = open("./"+args.save+'/'+"a_info.txt", "w")
    for i in info:
        f.write(i)
        f.write('\n')
    f.close()
    
def information(args):
    info = []
    info.append('======================================')
    info.append(f'    IID: {args.iid}')
    info.append(f'    p: {args.p}')
    info.append(f'    Dataset: {args.dataset}')
    info.append(f'    Model: {args.model}')
    info.append(f'    Model Init: {args.init}')
    info.append(f'    Aggregation Function: {args.defence}')
    info.append(f'    start: {args.attackT}')
    info.append(f'    fre: {args.frequency}')
    info.append(f'    trigger: {args.trigger}')
    info.append(f'    trigger_eopch: {args.local_ep_trigger}')
    info.append(f'    ifour {args.ifour}')
    if math.isclose(args.malicious, 0) == False:
        info.append(f'    Attack method: {args.attack}')
        if 'adaptive' in args.attack:
            info.append(f'    Attack mode: {args.ada_mode}')
        info.append(f'    Attack tau: {args.tau}')
        info.append(f'    Fraction of malicious agents: {args.malicious*100}%')
        info.append(f'    Poison Frac: {args.poison_frac}')
        info.append(f'    Backdoor From {args.attack_goal} to {args.attack_label}')
        info.append(f'    Attack Begin: {args.attack_begin}')
        info.append(f'    Trigger Shape: {args.trigger}')
        if args.trigger == 'square' or args.trigger == 'pattern':
            info.append(f'    Trigger Position X: {args.triggerX}')
            info.append(f'    Trigger Position Y: {args.triggerY}')
        
    else:
        info.append(f'    -----No Attack-----')
        
    info.append(f'    Number of agents: {args.num_users}')
    info.append(f'    Fraction of agents each turn: {int(args.num_users*args.frac)}({args.frac*100}%)')
    info.append(f'    Local batch size: {args.local_bs}')
    info.append(f'    Local epoch: {args.local_ep}')
    info.append(f'    local ep mal: {args.local_ep_mal}')
    info.append(f'    Client_LR: {args.lr}')
    info.append(f'    mal_LR: {args.lr_mal}')
    # print(f'    Server_LR: {args.server_lr}')
    info.append(f'    Client_Momentum: {args.momentum}')
    info.append(f'    Global Rounds: {args.epochs}')
    if args.defence == 'rlr':
        info.append(f'    RobustLR_threshold: {args.robustLR_threshold}')
    elif args.defence == 'fltrust' or args.defence == 'fltrust_bn':
        info.append(f'    Dataset In Server: {args.server_dataset}')
    info.append('======================================')
    return info

def get_base_info(args):
    if args.defence == 'RLR':
         base_info = '{}_{}_{}_{}_{}_Bg{}_F{}_Ag{}_mep{}_BN{}_scale{}_p{}_iid{}'.format(args.dataset,
                args.model, args.defence, args.robustLR_threshold, int(time.time()),args.attackT, args.frequency, args.alignUpdate,args.local_ep_mal,args.BN,args.scale_int,args.p,args.iid)
    else:
        base_info = '{}_{}_{}_{}_Bg{}_F{}_Ag{}_mep{}_BN{}_scale{}_p{}_iid{}_model{}'.format(args.dataset,
                    args.model, args.defence,  int(time.time()), args.attackT, args.frequency, args.alignUpdate, args.local_ep_mal,args.BN,args.scale_int,args.p,args.iid,
                                                                                            args.integrate)
    if math.isclose(args.malicious, 0) == False:
        base_info = base_info + '_{}_{}malicious_{}poisondata'.format(args.attack, args.malicious, args.poison_frac)
        if 'adaptive' in args.attack:
            base_info += '_mode{}'.format(args.ada_mode)
    else:
        base_info = base_info + '_no_malicious'
    # 룰베이스/노이즈 설정을 파일명에 박아 ablation 셀을 구분 (timestamp만으론 식별 불가)
    if getattr(args, 'pb_rule', 0):
        base_info += '_rule-{}_d{}k{}'.format(
            getattr(args, 'pb_rule_dets', 'relmat,flip').replace(',', '-'),
            getattr(args, 'pb_rule_drop', 0.0), getattr(args, 'pb_rule_keep', 0.5))
    if getattr(args, 'noise', 0):
        base_info += '_n{}'.format(args.noise)
    return base_info