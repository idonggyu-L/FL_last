#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import pickle

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy
import math
from skimage import io
# import cv2
from skimage import img_as_ubyte
import os
# print(os.getcwd())
from client.AttackerUtils import get_attack_layers_no_acc, get_malicious_info, get_malicious_info_local
from client.add_trigger import add_trigger
from models.Nets import NarrowResNet18
from models.subnetutils import replace_Conv2d, replace_BatchNorm2d
from client.test import get_edge


# from utils.loadData import load_saved_dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalMaliciousUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, attack=None, order=None, malicious_list=None, dataset_test=None,
                 prev_update=None, edge=None):
        self.prev_update = prev_update
        # print(prev_update)
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(
            dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        if args.attack == 'edge':
            self.ldr_train = edge
            # clean_train, self.clean_loader = load_saved_dataset('clean_train.pkl')
            print('ok')
        if args.local_dataset == 1:
            self.args.data = DatasetSplit(dataset, idxs)
        
        # backdoor task is changing label from attack_goal to attack_label
        self.attack_label = args.attack_label
        self.attack_goal = args.attack_goal
        
        self.model = args.model
        self.poison_frac = args.poison_frac
        if attack is None:
            self.attack = args.attack
        else:
            self.attack = attack

        self.trigger = args.trigger
        self.triggerX = args.triggerX
        self.triggerY = args.triggerY
        self.watermark = None
        self.apple = None
        self.dataset = args.dataset
        self.args.save_img = self.save_img
        if self.attack == 'dba':
            self.args.dba_class = int(order % 4)
        elif self.attack == 'get_weight':
            self.idxs = list(idxs)

        if malicious_list is not None:
            self.malicious_list = malicious_list
        if dataset is not None:
            self.dataset_train = dataset
        if dataset_test is not None:
            self.dataset_test = dataset_test
            
    def add_trigger(self, image):
        return add_trigger(self.args, image)

    
            
    def trigger_data(self, images, labels):
        #  attack_goal == -1 means attack all label to attack_label
        if self.attack_goal == -1:
            if math.isclose(self.poison_frac, 1):  # 100% copy poison data
                bad_data, bad_label = copy.deepcopy(
                        images), copy.deepcopy(labels)
                for xx in range(len(bad_data)):
                    bad_label[xx] = self.attack_label
                    # bad_data[xx][:, 0:5, 0:5] = torch.max(images[xx])
                    bad_data[xx] = self.add_trigger(bad_data[xx])
                images = torch.cat((images, bad_data), dim=0)
                labels = torch.cat((labels, bad_label))
            else:
                for xx in range(len(images)):  # poison_frac% poison data
                    labels[xx] = self.attack_label
                    # images[xx][:, 0:5, 0:5] = torch.max(images[xx])
                    images[xx] = self.add_trigger(images[xx])
                    if xx > len(images) * self.poison_frac:
                        break
        else:  # trigger attack_goal to attack_label
            if math.isclose(self.poison_frac, 1):  # 100% copy poison data
                bad_data, bad_label = copy.deepcopy(
                        images), copy.deepcopy(labels)
                for xx in range(len(bad_data)):
                    if bad_label[xx]!= self.attack_goal:  # no in task
                        continue  # jump
                    bad_label[xx] = self.attack_label
                    bad_data[xx] = self.add_trigger(bad_data[xx])
                    images = torch.cat((images, bad_data[xx].unsqueeze(0)), dim=0)
                    labels = torch.cat((labels, bad_label[xx].unsqueeze(0)))
            else:  # poison_frac% poison data
                # count label == goal label
                num_goal_label = len(labels[labels==self.attack_goal])
                counter = 0
                for xx in range(len(images)):
                    if labels[xx] != 0:
                        continue
                    labels[xx] = self.attack_label
                    # images[xx][:, 0:5, 0:5] = torch.max(images[xx])
                    images[xx] = self.add_trigger(images[xx])
                    counter += 1
                    if counter > num_goal_label * self.poison_frac:
                        break
        return images, labels
        
    def train(self, net, test_img = None, args=None):
        if self.attack == 'badnet':
            return self.train_malicious_badnet(net)
        elif self.attack == 'edge':
            return self.train_malicious_edge(net)
        elif self.attack == 'LGA':
            return self.train_ours(net,test_img=test_img, dataset_test=self.dataset_test, args=args)
        elif self.attack == 'neu':
            return self.train_neu(net,test_img=test_img, dataset_test=self.dataset_test, args=args)
        elif self.attack == 'AFA':
            return self.train_malicious_flipupdate(net)
        elif self.attack == 'LFA':
            return self.train_malicious_LFA(net)
        elif self.attack == 'dba':
            return self.train_malicious_dba(net)
        elif self.attack == "LPA":
            return self.train_malicious_LPA(net)
        elif self.attack == "adaptive":
            return self.train_malicious_adaptive(net)
        elif self.attack == "adaptive_local":
            return self.train_malicious_adaptive_local(net)
        elif self.attack == 'subnet':
            return self.train_malicious_subnet(net)
        elif self.attack == 'daa':
            return self.distance_awareness_attack(net)
        elif self.attack == 'daa2':
            # 1. train a benign model
            # 2. regularize on benign model to train malicious model
            # 3. main task loss should be the same as benign model
            # return self.distance_awareness_attack2(net)
            return self.distance_awareness_attack2(net)
        elif self.attack == 'scaling':
            return self.train_scaling_attack(net)
        else:
            print("Error Attack Method")
            os._exit(0)

    

    
    def regularization_loss(self, model1, model2):
        loss = 0
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            loss += torch.mean(torch.pow(param1 - param2, 2))
        return loss
        
    
    def distance_awareness_attack(self, net, test_img=None, dataset_test=None, args=None):
        # regularize distance and make it similar to the global model in the previous round
        previous_global_model = copy.deepcopy(net)
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = self.trigger_data(images, labels)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = (1-self.args.beta) * self.loss_func(log_probs, labels) + self.args.beta * self.regularization_loss(previous_global_model, net)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def distance_awareness_attack2(self, net, test_img=None, dataset_test=None, args=None):
        # train a benign model as the reference model
        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        # train malicious model under regularization
        malicious_model = copy.deepcopy(net)
        malicious_model.train()
        optimizer = torch.optim.SGD(
            malicious_model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        print('maliciousupdate.py',self.args.beta)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = self.trigger_data(images, labels)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                malicious_model.zero_grad()
                log_probs = malicious_model(images)
                loss = (1-self.args.beta) * self.loss_func(log_probs, labels) + self.args.beta * self.regularization_loss(malicious_model, net)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        bad_net_param = malicious_model.state_dict()
        return bad_net_param, sum(epoch_loss) / len(epoch_loss)
    


    
    def train_malicious_LFA(self, net, test_img=None, dataset_test=None, args=None):
        good_param = copy.deepcopy(net.state_dict())
        badnet = copy.deepcopy(net)
        badnet.train()
        # train and update
        optimizer = torch.optim.SGD(
            badnet.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                bad_data, bad_label = copy.deepcopy(
                    images), copy.deepcopy(labels)
                images, labels = self.trigger_data(images, labels)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                badnet.zero_grad()
                log_probs = badnet(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        bad_net_param = badnet.state_dict()
        self.malicious_model = copy.deepcopy(badnet)
       
        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        self.benign_model = copy.deepcopy(net)
        attack_param = {}
        attack_list = get_attack_layers_no_acc(copy.deepcopy(net.state_dict()), self.args)
        
        for layer in self.args.attack_layers:
            if layer not in attack_list:
                attack_list.append(layer)
        print(attack_list)
        for key, var in net.state_dict().items():
            if key in attack_list:
                difference = (bad_net_param[key]-good_param[key])
                attack_param[key] = good_param[key] - difference
            else:
                attack_param[key] = var
        return attack_param, sum(epoch_loss) / len(epoch_loss), attack_list

    
    def train_malicious_LPA(self, net, test_img=None, dataset_test=None, args=None):
        good_param = copy.deepcopy(net.state_dict())
        badnet = copy.deepcopy(net)
        badnet.train()
        # train and update
        optimizer = torch.optim.SGD(
            badnet.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                bad_data, bad_label = copy.deepcopy(
                    images), copy.deepcopy(labels)
                images, labels = self.trigger_data(images, labels)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                badnet.zero_grad()
                log_probs = badnet(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        bad_net_param = badnet.state_dict()
        self.malicious_model = copy.deepcopy(badnet)
       
        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        self.benign_model = copy.deepcopy(net)
        attack_param = {}
        attack_list = get_attack_layers_no_acc(copy.deepcopy(net.state_dict()), self.args)
        
        print('MaliciousUpdate line451 attack_list:',attack_list)
        for key, var in net.state_dict().items():
            if key in attack_list:
                difference = (bad_net_param[key]-good_param[key])
                x = 1
                attack_param[key] = good_param[key] + x * difference
            else:
                attack_param[key] = var
        return attack_param, sum(epoch_loss) / len(epoch_loss), attack_list

    def train_malicious_adaptive(self, net):
        global_param = copy.deepcopy(net.state_dict())
        badnet = copy.deepcopy(net)
        badnet.train()
        # train and update
        optimizer = torch.optim.SGD(
            badnet.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                bad_data, bad_label = copy.deepcopy(
                    images), copy.deepcopy(labels)
                images, labels = self.trigger_data(bad_data, bad_label)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                badnet.zero_grad()
                log_probs = badnet(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        bad_net_param = badnet.state_dict()
        self.malicious_model = copy.deepcopy(badnet)

        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        malicious_info = get_malicious_info(global_param, self.args, dataset_train=self.dataset_train, dataset_test=self.dataset_test)
        malicious_info['local_malicious_model'] = bad_net_param
        malicious_info['local_benign_model'] = net.state_dict()
        '''
        malicious_info{
        key_arr:
        value_arr:
        local_malicious_model:
        local_benign_model
        malicious_model_BSR:
        mal_val_dataset:
        }
        '''
        return sum(epoch_loss) / len(epoch_loss), malicious_info
    
    def train_malicious_adaptive_local(self, net):
        global_param = copy.deepcopy(net.state_dict())
        badnet = copy.deepcopy(net)
        badnet.train()
        # train and update
        optimizer = torch.optim.SGD(
            badnet.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                bad_data, bad_label = copy.deepcopy(
                    images), copy.deepcopy(labels)
                images, labels = self.trigger_data(bad_data, bad_label)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                badnet.zero_grad()
                log_probs = badnet(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        bad_net_param = badnet.state_dict()
        self.malicious_model = copy.deepcopy(badnet)

        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        self.benign_model = copy.deepcopy(net)
        malicious_info = get_malicious_info_local(self.benign_model, self.malicious_model, self.args, dataset_train=self.dataset_train, dataset_test=self.dataset_test)
        malicious_info['local_malicious_model'] = bad_net_param
        malicious_info['local_benign_model'] = net.state_dict()
        '''
        malicious_info{
        key_arr:
        value_arr:
        local_malicious_model:
        local_benign_model
        malicious_model_BSR:
        mal_val_dataset:
        }
        '''
        return sum(epoch_loss) / len(epoch_loss), malicious_info

    def train_malicious_badnet(self, net, test_img=None, dataset_test=None, args=None):
        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr_mal, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep_mal):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = self.trigger_data(images, labels)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_malicious_dba(self, net, test_img=None, dataset_test=None, args=None):
        global_param = copy.deepcopy(net.state_dict())
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep_mal):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = self.trigger_data(images, labels)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)



    def train_benign(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def save_img(self, image):
        img = image
        if image.shape[0] == 1:
            pixel_min = torch.min(img)
            img -= pixel_min
            pixel_max = torch.max(img)
            img /= pixel_max
            io.imsave('./save/backdoor_trigger.png', img_as_ubyte(img.squeeze().numpy()))
        else:
            img = image.numpy()
            img = img.transpose(1, 2, 0)
            pixel_min = np.min(img)
            img -= pixel_min
            pixel_max = np.max(img)
            img /= pixel_max
            if self.attack == 'dba':
                io.imsave('./save/dba'+str(self.args.dba_class)+'_trigger.png', img_as_ubyte(img))
            io.imsave('./save/backdoor_trigger.png', img_as_ubyte(img))


    def train_ours(self, net, test_img=None, dataset_test=None, args=None):
        global_param = copy.deepcopy(net.state_dict())
        bad_net = copy.deepcopy(net)
        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr*0.998**(self.args.iter), momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        bad_net.train()
        lr = self.args.lr_mal
        if self.args.iter > 900:
            lr /= 2
        # train and update
        optimizer = torch.optim.SGD(
            bad_net.parameters(), lr=lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep_mal):
            self.local_ep = iter
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = self.trigger_data(images, labels)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                bad_net.zero_grad()
                log_probs = bad_net(images)
                loss = (self.loss_func(log_probs, labels))
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            new = self.getNewW(global_param, bad_net.state_dict(), net.state_dict())

            bad_net.load_state_dict(new)

        if self.args.iter >= 0 and (self.args.iter % 20 == 0 or self.args.iter >= 280):
            print(self.args.iter)
            acc_test, _, backdoor_acc = test_img(
                copy.deepcopy(bad_net), dataset_test, self.args, test_backdoor=True)
            args.local_bsr.append(backdoor_acc)
            # print(args.local_bsr,'local bsr')
            print('backdoor attack')
            print("local Testing accuracy: {:.2f}".format(acc_test))
            print("local Backdoor accuracy: {:.2f}".format(backdoor_acc))

        return bad_net.state_dict(), sum(epoch_loss) / len(epoch_loss)




    def get_loss(self,glob_w, new_w):
        norm_g = self.parameters_dict_to_vector_flt(self.prev_update)
        bad_ft = self.parameters_dict_to_vector_flt(new_w)
        glob_ft = self.parameters_dict_to_vector_flt(glob_w)
        return (bad_ft-glob_ft-norm_g).norm()


    def getNewW(self, glob_w, new_w, clean_w, output=False):
        mask ={}
        layers = [key for key in new_w.keys() if 'num_batches_tracked' not in key]
        layer_norm = []
        for idx in new_w.keys():
            if 'num_batches_tracked' in idx :
                new_w[idx] = glob_w[idx]
                continue
            layer_name = idx
            params = glob_w[layer_name]
            norm_origin = (clean_w[layer_name] - params).norm()
            self.args.M = (self.prev_update[layer_name]).norm()
            layer_norm.append([torch.norm(new_w[layer_name] - params).item(),layer_name])
            scale = min(1, self.args.M/(torch.norm(new_w[layer_name] - params).item() + 1e-8) )
            new_w[layer_name] = (new_w[layer_name]-params) * scale + params
        return new_w

    def parameters_dict_to_vector_flt(self, net_dict) -> torch.Tensor:
        vec = []
        for key, param in net_dict.items():
            # if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
            #     continue
            if key.split('.')[-1] == 'num_batches_tracked':
                continue
            vec.append(param.view(-1))
        return torch.cat(vec)

    def parameters_dict_to_vector_fltno(self, net_dict) -> torch.Tensor:
        vec = []
        for key, param in net_dict.items():
            if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
                continue
            # if key.split('.')[-1] == 'num_batches_tracked':
            #     continue
            vec.append(param.view(-1))
        return torch.cat(vec)




class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.data = data_dict['data']
        self.targets = data_dict['targets']
        self.transform = data_dict['transform']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, target


def load_saved_dataset(filename, batch_size=64):
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f)
    dataset = CustomDataset(data_dict)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=('train' in filename))
    return dataset, loader