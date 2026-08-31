import copy

import numpy as np
import torch
from models.Nets import get_model, ResNet18, vgg11, vgg19_bn




class Multi_metrics(object):   #Our defense

    def __init__(self, num_workers, num_adv, args):
        self.num_workers = num_workers
        self.s = num_adv
        self.args = args

    def exec(self, client_models, num_dps):
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        numC = len(vectorize_nets)
        cos_dis = [0.0] * numC
        length_dis = [0.0] * numC
        manhattan_dis = [0.0] * numC
        for i, g_i in enumerate(vectorize_nets):
            for j in range(len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]
                    cosine_distance = float(
                        (1 - np.dot(g_i, g_j) / (np.linalg.norm(g_i) * np.linalg.norm(g_j))) ** 2)   #Compute the different value of cosine distance
                    manhattan_distance = float(np.linalg.norm(g_i - g_j, ord=1))    #Compute the different value of Manhattan distance
                    length_distance = np.abs(float(np.linalg.norm(g_i) - np.linalg.norm(g_j)))    #Compute the different value of Euclidean distance

                    cos_dis[i] += cosine_distance
                    length_dis[i] += length_distance
                    manhattan_dis[i] += manhattan_distance
        print('cos dis:', cos_dis)
        print(np.argsort(cos_dis))
        print('dis,', length_dis)
        print(np.argsort(length_dis))
        print('man,', manhattan_dis)
        print(np.argsort(manhattan_dis))
        self.args.mm1.append(cos_dis)
        self.args.mm2.append(length_dis)
        self.args.mm3.append(manhattan_dis)
        tri_distance = np.vstack([cos_dis, manhattan_dis, length_dis]).T

        cov_matrix = np.cov(tri_distance.T)
        inv_matrix = np.linalg.inv(cov_matrix)

        ma_distances = []
        for i, g_i in enumerate(vectorize_nets):
            t = tri_distance[i]
            ma_dis = np.dot(np.dot(t, inv_matrix), t.T)
            ma_distances.append(ma_dis)

        scores = ma_distances
        print(scores)


        p = 0.3
        p_num = p*len(scores)
        topk_ind = np.argpartition(scores, int(p_num))[:int(p_num)]   #sort
        print(topk_ind)

        # selected_num_dps = np.array(num_dps)[topk_ind]
        # reconstructed_freq = [snd / sum(selected_num_dps) for snd in selected_num_dps]

        # logger.info("Num data points: {}".format(num_dps))
        # logger.info("Num selected data points: {}".format(selected_num_dps))
        # logger.info("The chosen ones are users: {}, which are global users: {}".format(topk_ind,
        #                                                                                [g_user_indices[ti] for ti in
        #                                                                                 topk_ind]))
        # aggregated_grad = np.average(np.array(vectorize_nets)[topk_ind, :], weights=reconstructed_freq,
        #                              axis=0).astype(np.float32)
        #
        # aggregated_model = client_models[0]  # slicing which doesn't really matter
        # load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to(self.args.device))
        # neo_net_list = [aggregated_model]
        # logger.info("Norm of Aggregated Model: {}".format(torch.norm(torch.nn.utils.parameters_to_vector(aggregated_model.parameters())).item()))
        neo_net_freq = [1.0]
        return topk_ind, cos_dis, length_dis, manhattan_dis

    def exec_cuda(self, client_models):
        vectorize_nets = [vectorize_net(cm) for cm in client_models]
        vectorize_nets_np = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        numC = len(vectorize_nets)
        cos_dis = [0.0] * numC
        length_dis = [0.0] * numC
        manhattan_dis = [0.0] * numC
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
        for i, g_i in enumerate(vectorize_nets):
            for j in range(len(vectorize_nets)):
                if i!= j:
                    g_j = vectorize_nets[j]
                    cosine_distance = 1 - cos(g_i, g_j)
                    manhattan_distance = torch.norm(g_i - g_j, p=1)
                    length_distance = torch.abs(torch.norm(g_i) - torch.norm(g_j))
                    cos_dis[i] += cosine_distance.item()
                    length_dis[i] += length_distance.item()
                    manhattan_dis[i] += manhattan_distance.item()

        print('cos dis:',cos_dis)
        print( np.argsort(cos_dis))
        print('dis,',length_dis)
        print(np.argsort(length_dis))
        print('man,',manhattan_dis)
        print(np.argsort(manhattan_dis))
        self.args.mm1.append(cos_dis)
        self.args.mm2.append(length_dis)
        self.args.mm3.append(manhattan_dis)
        tri_distance = np.vstack([cos_dis, manhattan_dis, length_dis]).T

        cov_matrix = np.cov(tri_distance.T)
        inv_matrix = np.linalg.inv(cov_matrix)
        ma_distances = []
        for i, g_i in enumerate(vectorize_nets):
            t = tri_distance[i]
            ma_dis = np.dot(np.dot(t, inv_matrix), t.T)
            ma_distances.append(ma_dis)
        scores = ma_distances
        print(scores)
        print(np.argsort(scores))

        p = 0.3
        p_num = p * len(scores)
        topk_ind = np.argpartition(scores, int(p_num))[:int(p_num)]  # sort
        print(topk_ind)
        num_clients = max(int(self.args.frac * self.args.num_users), 1)
        num_malicious_clients = int(self.args.malicious * num_clients)

        for i in range(len(topk_ind)):
            if topk_ind[i] < num_malicious_clients:
                self.args.wrong_mal += 1
            else:
                    #  minus per benign in cluster
                self.args.right_ben += 1
        self.args.turn += 1

            #
        # selected_num_dps = np.array(num_dps)[topk_ind]
        # reconstructed_freq = [snd / sum(selected_num_dps) for snd in selected_num_dps]
        #
        # aggregated_grad = np.average(np.array(vectorize_nets_np)[topk_ind, :], weights=reconstructed_freq,
        #                              axis=0).astype(np.float32)
        #
        # aggregated_model = client_models[0] # slicing which doesn't really matter
        # load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to(self.args.device))
        # neo_net_list = [aggregated_model]
        # # logger.info("Norm of Aggregated Model: {}".format(torch.norm(torch.nn.utils.parameters_to_vector(aggregated_model.parameters())).item()))
        # neo_net_freq = [1.0]
        # netavg = fed_avg_aggregator(neo_net_list,neo_net_freq, self.args)

        return  topk_ind, cos_dis, length_dis, manhattan_dis


def get_new_model(args):
    if args.model == 'VGG' and args.dataset == 'cifar':
        net_glob = vgg19_bn().to(args.device)
    elif args.model == "resnet" and args.dataset == 'cifar':
        net_glob = ResNet18().to(args.device)
    elif args.model == "resnet" and args.dataset == 'cifar100':
        net_glob = ResNet18(num_classes=100).to(args.device)
    else:
        exit('Error: unrecognized model')
    return net_glob


def vectorize_net(net):
    # return torch.cat([p.view(-1) for p in net.parameters()])
    vec = []
    for key, param in net.state_dict().items():
        # if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
        #     continue
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)
def load_model_weight(net, weight):
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data = weight[index_bias:index_bias + p.numel()].view(p.size())
        index_bias += p.numel()

def fed_avg_aggregator(net_list, net_freq, args):
    # net_avg = VGG('VGG11').to(device)
    net_avg = get_new_model(args)
    whole_aggregator = []

    for p_index, p in enumerate(net_list[0].parameters()):
        # initial
        params_aggregator = torch.zeros(p.size()).to(args.device)
        for net_index, net in enumerate(net_list):
            # we assume the adv model always comes to the beginning
            params_aggregator = params_aggregator + net_freq[net_index] * list(net.parameters())[p_index].data
        whole_aggregator.append(params_aggregator)

    for param_index, p in enumerate(net_avg.parameters()):
        p.data = whole_aggregator[param_index]
    return net_avg