import copy

import numpy as np
import torch

from client.Update import LocalUpdate


def flare(w_updates, w_locals, net, central_dataset, dataset_test, global_parameters, args):
    w_feature = []
    temp_model = copy.deepcopy(net)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()

    for client in w_locals:
        net.load_state_dict(client)
        local = LocalUpdate(
            args=args, dataset=dataset_test, idxs=central_dataset)
        feature = local.get_PLR(
            net=copy.deepcopy(net).to(args.device))
        w_feature.append(feature)
    distance_list = [[] for i in range(len(w_updates))]
    # distance_list=[list(len(w_updates)) for i in range(len(w_updates))]
    for i in range(len(w_updates)):
        for j in range(i + 1, len(w_updates)):
            score = compute_mmd(w_feature[i], w_feature[j])
            distance_list[i].append(score.item())
            distance_list[j].append(score.item())
    # print('defense line121 distance_list', distance_list)
    vote_counter = [0 for i in range(len(w_updates))]
    k = round(len(w_updates) * 0.5)
    for i in range(len(w_updates)):
        IDs = np.argsort(distance_list[i])
        for j in range(len(IDs)):
            if IDs[j] >= i:
                client_id = IDs[j] + 1
            else:
                client_id = IDs[j]
            vote_counter[client_id] += 1
            if j + 1 >= k:  # first 𝑘 elements in 𝐼 𝐷𝑠 and vote for it
                break
    trust_score = [x / sum(vote_counter) for x in vote_counter]
    print('defense line188 len trust_score', trust_score)

    w_avg = copy.deepcopy(global_parameters)
    for k in w_avg.keys():
        for i in range(0, len(w_updates)):
            try:
                w_avg[k] += w_updates[i][k] * trust_score[i]
            except:
                w_updates[i][k] = w_updates[i][k].type_as(w_avg[k]).long()
                w_avg[k] = w_avg[k].long() + w_updates[i][k] * trust_score[i]
    return w_avg

def compute_mmd(x, y):
    # Compute the MMD between two tensors x and y (RBF kernel).
    # 벡터화 버전: torch.cdist로 전체 커널행렬을 한 번에 계산. 기존 이중루프와 수치적으로 동일
    # (대각 포함 합 / m(m-1) 등). python 루프 대비 수십~수백배 빠름.
    sigma = 1.0
    m = x.size(0)
    n = y.size(0)

    def _rbf(a, b):
        d2 = torch.cdist(a, b) ** 2          # (|a|, |b|) squared L2
        return torch.exp(-d2 / (2 * sigma ** 2))

    xx_kernel = _rbf(x, x)                    # 대각=1 포함 (기존과 동일)
    yy_kernel = _rbf(y, y)
    xy_kernel = _rbf(x, y)
    mmd = (torch.sum(xx_kernel) / (m * (m - 1))) + (torch.sum(yy_kernel) / (n * (n - 1))) - (2 * torch.sum(xy_kernel) / (m * n))
    return mmd

def kernel_function(x, y):
    sigma = 1.0
    return torch.exp(-torch.norm(x - y) ** 2 / (2 * sigma ** 2))