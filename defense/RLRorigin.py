import copy

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class Rlr():
    def __init__(self, args):
        self.args = args
        self.server_lr = args.server_lr
        self.cum_net_mov = 0

    def aggregate_updates(self, global_model, params, net_list):
        agent_updates_dict = {}
        global_param = parameters_to_vector(global_model.parameters())
        for i,net in enumerate(net_list):
            net_params = parameters_to_vector(net.parameters())
            net_updates = net_params - global_param
            agent_updates_dict[i] = net_updates
        n_params = len(agent_updates_dict[0])
        # adjust LR if robust LR is selected
        lr_vector = torch.Tensor([self.server_lr] * n_params).to(self.args.device)
        lr_vector = self.compute_robustLR(agent_updates_dict)
        aggregated_updates = self.agg_avg(agent_updates_dict)

        cur_global_params = parameters_to_vector(global_model.parameters())
        new_global_params = (cur_global_params + lr_vector * aggregated_updates).float()
        aggregated_model = copy.deepcopy(net_list[-1])
        aggregated_model.load_state_dict(params)
        self.load_model_weight( aggregated_model, new_global_params)
        return aggregated_model.state_dict()



    def compute_robustLR(self, agent_updates_dict):
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        sm_of_signs[sm_of_signs < self.args.robustLR_threshold] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.args.robustLR_threshold] = self.server_lr
        return sm_of_signs.to(self.args.device)

    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """
        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = 1
            sm_updates += n_agent_data * update
            total_data += n_agent_data
        return sm_updates / total_data

    # def parameters_to_vector(self, model):
    def vectorize_net(self, net):
        vec = []
        for key, param in net.items():
            if key.split('.')[-1] == 'num_batches_tracked':
                continue
            vec.append(param.view(-1))
        return torch.cat(vec)

    def load_model_weight(self, net, weight):
        index_bias = 0
        for p_index, p in enumerate(net.parameters()):
            p.data = weight[index_bias:index_bias + p.numel()].view(p.size())
            index_bias += p.numel()

    def load_model_weight2(self, net, weight):
        index_bias = 0
        for p_index, p in (net.state_dict().items()):
            if p_index.split('.')[-1] == 'num_batches_tracked' :
                continue
            # print(p)
            p = weight[index_bias:index_bias + p.numel()].view(p.size())
            # print(p)
            index_bias += p.numel()
        print(index_bias,len(weight))

