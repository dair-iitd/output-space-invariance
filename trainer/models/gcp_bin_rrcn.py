import math
import pickle
from IPython.core.debugger import Pdb
import copy
import collections
import functools
import os
import json
from collections import Counter 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import random as py_random
import jacinle.random as random
import jacinle.io as io
import jactorch.nn as jacnn

from jactorch.utils.meta import as_tensor, as_float, as_cpu
from difflogic.thutils_rl import binary_accuracy, instance_accuracy
from difflogic.train import TrainerBase

from jacinle.logging import get_logger, set_output_file
from jacinle.utils.container import GView
from jacinle.utils.meter import GroupMeters
from jactorch.utils.meta import as_cuda, as_numpy, as_tensor
import time
import datetime

import dgl
from .rrn.gcp_bin_cluster import GCP_BIN_CNN

import utils

logger = get_logger(__file__)


class GCP_BIN_RRCN(nn.Module):
    def __init__(self, args):
        super(GCP_BIN_RRCN, self).__init__()
        self.args = args
        self.num_steps = args.msg_passing_steps
        self.gcp_solver = GCP_BIN_CNN(args = args) 

        self.wt_base_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        
        
        def loss_aggregator(pred, target_in, target_mask, mask_of_num_nodes, mask_of_num_colours):
            # Accouting for difference in number of nodes for gcp
            batch_num_nodes = pred.shape[2]
            target = target_in[:,:,:batch_num_nodes]
            
            batch_size, target_size, num_variables = target.size()
            num_steps = pred.size(-1)

            pred = pred.unsqueeze(-1).expand(*pred.size(),target_size)
            
            target= target.transpose(1,2).unsqueeze(-1).expand(batch_size, num_variables, target_size,num_steps).transpose(-1,-2) 
            bin_target = torch.nn.functional.one_hot(target.long()).transpose(3,4).transpose(2,3).transpose(1,2)
            
            loss_tensor = self.wt_base_loss(pred[:,1:], bin_target.float()[:,1:])    # Ignore 0th color in loss to avoid underflow
            
            loss_tensor = loss_tensor.mean(dim=3)
            

            mask_nodes = mask_of_num_nodes.unsqueeze(1).unsqueeze(-1).expand_as(loss_tensor).float()
            loss_tensor = (loss_tensor*mask_nodes).sum(dim=2)/mask_nodes.sum(dim=2)
            
            mask_colours = mask_of_num_colours[:,1:].unsqueeze(-1).expand_as(loss_tensor).float()
            loss_tensor = (loss_tensor*mask_colours).sum(dim=1)/mask_colours.sum(dim=1)
            
            loss_tensor = loss_tensor*(target_mask.float()) 
            
            if self.args.min_loss:
                loss_tensor  = loss_tensor.masked_fill((target_mask<1),float('inf')).min(dim=1)[0]

            return loss_tensor
        self.loss = loss_aggregator
        # add to targetset is set externally from the trainer/train.py
        self.add_to_targetset = False

    def forward(self, feed_dict,return_loss_matrix = False, can_break = False,get_all_steps=False):
        
        feed_dict = GView(feed_dict)
       
        
        batch_size = feed_dict['query'].size(0)
        max_num_colours = feed_dict['chromatic_num'].max()
        num_colours_list = feed_dict['chromatic_num']
        max_num_nodes = feed_dict['num_nodes'].max()
        num_nodes_list = feed_dict['num_nodes']
        
        if self.args.logk: 
            max_levels = math.floor(math.log2(max_num_colours.item())) + 1
        else:
            max_levels = max_num_colours.item()
            

        mask_of_num_colours = torch.arange(max_num_colours+1, device = num_nodes_list.device)[None,:] <= num_colours_list[:,None]
        mask_of_num_nodes = torch.arange(num_nodes_list.max(), device = num_nodes_list.device)[None,:] < num_nodes_list[:,None] 
        
        t0 = time.time()
        #bg = self.collate_fn(feed_dict, self.have_may_edges, self.refine_diff_edges)
        bg = feed_dict['bg']
        t1 = time.time()
        
        time_collate = t1 - t0
        t0 = t1

        #Pdb().set_trace()
        #logits : of shape : args.num_steps x batchsize*(max_num_nodes) x (1+max_num_colours) if training
        #logits: of shape : batch_size* max_num_nodes x max_num_colours if not training
         
        ret_dict, bin_logits, rrcn_time  = self.gcp_solver(bg,batch_size, num_colours_list,num_nodes_list , self.training, get_all_steps = get_all_steps)
          

        t1 = time.time()
        time_solve = rrcn_time
        t0 = t1

        if self.training or get_all_steps:
            bin_logits = bin_logits.transpose(0,1)
        else:
            bin_logits = bin_logits.unsqueeze(-1)
        #shape of logits currently : (batch_size*num_nodes) x time_steps(1)
        
        bin_logits = bin_logits.view(-1, max_num_nodes*max_levels, bin_logits.size(-1)) 
            
        # Input shape: BS*(num_nodes*num_levels)*TS
        # Output shape: BS*(num_colors+1)*num_nodes*TS or BS*(num_levels)*num_nodes*TS
        bs, ts, num_nodes = bin_logits.shape[0], bin_logits.shape[-1], bin_logits.shape[1]//max_levels
        #dig_scores = bin_logits.reshape(bs, num_nodes, max_num_colours.item(), ts).transpose(1,2)
        #Pdb().set_trace()
        dig_scores = bin_logits.reshape(bs, max_levels, num_nodes, ts)
        
        if self.args.logk:
            
#             dig_rpt = torch.sigmoid(dig_scores.transpose(0,1).unsqueeze(0).repeat(max_num_colours+1,1,1,1,1))
            dig_rpt = dig_scores.transpose(0,1).unsqueeze(0).repeat(max_num_colours+1,1,1,1,1)
            
            mask = (torch.arange(max_num_colours+1).unsqueeze(-1) & 2**torch.arange(max_levels)) > 0
            mask = mask.to(dig_scores.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(dig_rpt) 
            
            
            # Do weighted mean here
#             logits = torch.where(mask, dig_rpt, 1-dig_rpt).prod(dim=1).transpose(0,1)
            logits = torch.where(mask, dig_rpt, -dig_rpt).mean(dim=1).transpose(0,1)
            pred = logits
            
        else:
            logits = torch.cat([float('-inf')*torch.ones(bs, 1, num_nodes, ts).to(dig_scores.device), dig_scores], dim=1)
            bin_pred, pred = bin_logits, logits

        if self.training or self.add_to_targetset:
            this_meters, ia_output_dict = instance_accuracy(feed_dict.target.float(), pred, return_float= False, feed_dict=feed_dict, task = self.args.task, args=self.args)
            reward,new_targets = ia_output_dict['reward'], ia_output_dict['new_targets'] 
            
            if self.add_to_targetset:
                utils.add_missing_target(feed_dict,new_targets,reward)

            monitors = dict()
            target = feed_dict.target.float()
            count = None
            #Pdb().set_trace()
            loss_matrix = None
            """
            if self.args.min_loss or 'weights' in feed_dict or return_loss_matrix:
                loss_matrix = self.loss(logits, feed_dict.target_set,feed_dict.mask)
            else:
                loss_matrix = self.loss(logits, target.unsqueeze(1),feed_dict.mask[:,0].unsqueeze(-1))
            """ 
            if self.args.min_loss or 'weights' in feed_dict or return_loss_matrix:
                loss_matrix = self.loss(logits, feed_dict.target_set,feed_dict.mask, mask_of_num_nodes, mask_of_num_colours)
            else:
                loss_matrix = self.loss(logits, target.unsqueeze(1),feed_dict.mask[:,0].unsqueeze(-1), mask_of_num_nodes, mask_of_num_colours)
            #Pdb().set_trace()
            if 'weights' in feed_dict:
                loss = (feed_dict.weights*loss_matrix).sum()/feed_dict.weights.sum()
            else:
                loss = loss_matrix.mean()
          
            t1 = time.time()
            time_remaining = t1 - t0

           
            monitors['time_collate'], monitors['time_solve'], monitors['time_remaining'] = torch.tensor([time_collate]), torch.tensor([time_solve]), torch.tensor([time_remaining])
            
            monitors.update(this_meters)
           
            ret_dict.update(dict(pred=pred,reward=reward,loss_matrix = loss_matrix))
            return loss, monitors, ret_dict
        else:
            ret_dict.update(dict(pred=pred))
            return ret_dict
 
