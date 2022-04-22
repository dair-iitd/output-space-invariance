import math
import pickle
import copy
import collections
import functools
import os
import json
import time
import datetime
import numpy as np
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import random as py_random
import jacinle.random as random
import jacinle.io as io
import jactorch.nn as jacnn

from jactorch.utils.meta import as_tensor, as_float, as_cpu
from jacinle.logging import get_logger, set_output_file
from jacinle.utils.container import GView
from jacinle.utils.meter import GroupMeters
from jactorch.utils.meta import as_cuda, as_numpy, as_tensor

from difflogic.cli import format_args
from difflogic.nn.neural_logic import LogicMachine, LogicInference, LogitsInference, LogicSoftmaxInference
from difflogic.nn.neural_logic.modules._utils import meshgrid_exclude_self
from difflogic.thutils_rl import binary_accuracy, instance_accuracy
from IPython.core.debugger import Pdb
import utils

logger = get_logger(__file__)
class NLMModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        # inputs
        self.args = args
        
        if args.task_is_futoshiki:
            binary_dim = 4
            unary_dim = 2
        elif args.task_is_sudoku or args.task_is_gcp:
            binary_dim = 2
            unary_dim = 2
        #
        self.feature_axis = 1
        input_dims = [0 for _ in range(args.nlm_breadth + 1)]
        input_dims[0] = 0
        input_dims[1] = unary_dim
        input_dims[2] = binary_dim
        self.features = LogicMachine.from_args(
            input_dims, args.nlm_attributes, args, prefix='nlm')

        output_dim = self.features.output_dims[self.feature_axis]
        target_dim = 1
        self.pred = LogicInference(output_dim, target_dim, [])

        # losses
        self.base_loss = nn.BCELoss()
        self.wt_base_loss = nn.BCELoss(reduction='none')

        def loss_aggregator(pred, target, count, weights=None):
            # if pred and target have same dimension then simply compute loss
            # Pdb().set_trace()
            if pred.dim() == target.dim():
                # if weights are not none then weigh each datapoint appropriately else simply average them
                if weights is not None:
                    loss = (weights*self.wt_base_loss(pred,
                                                      target).sum(dim=1)).sum()/weights.sum()
                    return loss
                return self.base_loss(pred, target)

            # if pred and target are not of same dimension then compute loss wrt each element in target set
            # return a (batchsize x targetset size) vector
            batch_loss = []
            for i in range(len(pred)):
                x = pred[i]
                instance_loss = []
                for y in target[i][:count[i]]:
                    instance_loss.append(self.base_loss(x, y))
                if self.args.min_loss:
                    batch_loss.append(
                        torch.min(torch.stack(instance_loss)))
                elif self.args.naive_pll_loss:
                    batch_loss.append(torch.mean(
                        torch.stack(instance_loss)))
                else:
                    batch_loss.append(
                        F.pad(torch.stack(instance_loss), (0, len(target[i])-count[i]), "constant", 0))
            return torch.stack(batch_loss)

        self.loss = loss_aggregator

    def distributed_pred(self, inp, depth):
        feature = self.features(inp, depth=depth)[self.feature_axis]
        pred = self.pred(feature)
        pred = pred.squeeze(-1)
        return pred

    #def forward(self, feed_dict, return_loss_matrix=False):
    def forward(self, feed_dict,return_loss_matrix = False, can_break = False,get_all_steps=False):
        #Pdb().set_trace()
        feed_dict = GView(feed_dict)
        #board_size = int(math.sqrt(feed_dict['n'][0]))
        num_nodes = feed_dict['num_nodes'][0]
        num_classes = feed_dict['chromatic_num'][0]

        states = None

        # relations
        relations = feed_dict.nlm_relations.float()
        states = feed_dict.nlm_query.float()
        batch_size, nr = relations.size()[:2]
        inp = [None for _ in range(self.args.nlm_breadth + 1)]

        inp[1] = states
        inp[2] = relations
        depth = None
        if self.args.nlm_recursion:
            depth = 1
            while 2**depth + 1 < nr:
                depth += 1
            depth = depth * 2 + 1
        #logger.info(feed_dict['num_nodes'])
        bin_logits = self.distributed_pred(inp, depth=depth)
        dig_scores = bin_logits.reshape(batch_size, num_nodes, num_classes).transpose(1,2).unsqueeze(-1)
        logits = torch.cat([float('-inf')*torch.ones(batch_size, 1, num_nodes, 1).to(dig_scores.device), dig_scores], dim=1)
        bin_pred, pred = bin_logits, logits
        #   
        #Pdb().set_trace()
        #pw = ((bin_logits > 0.5).float() == feed_dict['nlm_target'].float()).float().mean()
        #logger.info("pw: {}".format(pw))


        #
        #Pdb().set_trace()
        if self.training:
            monitors = dict()
            this_meters, ia_output_dict = instance_accuracy(feed_dict.target.float(), pred, return_float= False, feed_dict=feed_dict, task = self.args.task, args=self.args)
            reward,new_targets = ia_output_dict['reward'], ia_output_dict['new_targets'] 
            if self.add_to_targetset:
                utils.add_missing_target(feed_dict,new_targets,reward)
            
            monitors.update(this_meters)
            
            target = feed_dict.nlm_target
            target = target.float()
            count = feed_dict['count'].int()
            if  self.args.min_loss or 'weights' in feed_dict or return_loss_matrix:
                target = feed_dict.nlm_target_set
                target = target.float()
                count = feed_dict.count.int()
            
            
            loss_matrix = self.loss(bin_logits, target, count)

            if self.args.min_loss:
                loss = loss_matrix.mean()
            elif 'weights' in feed_dict:
                loss = (feed_dict.weights*loss_matrix).sum() / \
                    feed_dict.weights.sum()
            else:
                loss = loss_matrix

            return loss, monitors, dict(pred=pred, reward=reward, loss_matrix=loss_matrix)
        else:
            return dict(pred=pred)

