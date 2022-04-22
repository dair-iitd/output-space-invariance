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
from .rrn.futo_bin_cluster import Futoshiki_BIN_CNN

import utils

logger = get_logger(__file__)


class Futoshiki_BIN_RRCN(nn.Module):
    def __init__(self, args):
        super(Futoshiki_BIN_RRCN, self).__init__()
        self.args = args
        self.num_steps = args.msg_passing_steps
        self.futo_solver = Futoshiki_BIN_CNN(args = args) 
        self.wt_base_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        
        
        def loss_aggregator(pred, target, target_mask):
            
            # if pred and target have same dimension then simply compute loss
            # pred.shape == BS x 10 x 81 x  32
            #target.shape = BS X Target size x 81
            #if pred.dim()==self.args.sudoku_num_steps*target.dim():
            #    return self.base_loss(pred, target)
            # if pred and target are not of same dimension then compute loss wrt each element in target set
            # return a (batchsize x targetset size) vector 
            #target.shape = batch_size x target size x num_variables
            #pred.shape: batch_size x num variables
            batch_size, target_size, num_variables = target.size()
            num_steps = pred.size(-1)

            #mask_of_num_colours.shape = BS x (max_num_colours+1)
            #pred.shape = BS x (max_num_colours+1) x max_num_nods x num_steps
            #target.shape = BS x TS x max_num_nodes
            
            pred = pred.unsqueeze(-1).expand(*pred.size(),target_size)
            
            target= target.transpose(1,2).unsqueeze(-1).expand(batch_size, num_variables, target_size,num_steps).transpose(-1,-2) 
#             Pdb().set_trace()
            bin_target = torch.nn.functional.one_hot(target.long()).transpose(3,4).transpose(2,3).transpose(1,2)
            loss_tensor = self.wt_base_loss(pred[:,1:], bin_target.float()[:,1:])    # Ignore 0th color in loss to avoid underflow
            
            # take average across mp steps, and puzzles
            loss_tensor = loss_tensor.mean(dim=[2,3,4])
            
            #shape = batch_size x target_size
            if self.args.min_loss:
                #return has shape: batch_size 
                loss_tensor  = loss_tensor.masked_fill((target_mask<1),float('inf')).min(dim=1)[0]

            return loss_tensor
        self.loss = loss_aggregator
        #self.loss = loss_aggregator_sudoku_rrcn
        self.add_to_targetset = False

   
    def forward(self, feed_dict,return_loss_matrix = False, can_break = False,get_all_steps=False):
       
    
        feed_dict = GView(feed_dict)
        t0 = time.time()
        batch_size = feed_dict['query'].size(0)
    
        #bg = self.collate_fn(feed_dict, self.have_may_edges, self.refine_diff_edges)
        bg = feed_dict['bg']
        board_size = round((bg.nodes('cell').shape[0]//batch_size)**(1./3))
        ret_dict, bin_logits, rrcn_time  = self.futo_solver(feed_dict, batch_size, self.training, get_all_steps = get_all_steps)
          

        t1 = time.time()
        time_collate, time_solve = 0., rrcn_time
        t0 = t1
        
        if self.training or get_all_steps:
            bin_logits = bin_logits.transpose(0,1)
        else:
            bin_logits = bin_logits.unsqueeze(-1)
        #shape of logits currently : (batch_size*num_nodes) x time_steps(1)
        ts = bin_logits.shape[-1]
        
        
        bin_logits = bin_logits.view(batch_size, board_size**3, ts) 
            
        # Input shape: BS*(num_cells*num_digits)*TS
        # Output shape: BS*(num_colors+1)*num_nodes*TS
        dig_scores = bin_logits.reshape(batch_size, board_size, board_size**2, -1)
        logits = torch.cat([float('-inf')*torch.ones(batch_size, 1, board_size**2, ts).to(dig_scores.device), dig_scores], dim=1)
        
        bin_pred, pred = bin_logits, logits

        if self.training or self.add_to_targetset:
            this_meters, ia_output_dict = instance_accuracy(feed_dict.target.float(), pred, return_float= False, feed_dict=feed_dict, task = self.args.task, args=self.args)
            reward,new_targets = ia_output_dict['reward'], ia_output_dict['new_targets'] 
          
            
            if self.add_to_targetset:
                #Pdb().set_trace()
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
                loss_matrix = self.loss(logits, feed_dict.target_set, feed_dict.mask)
            else:
                loss_matrix = self.loss(logits, target.unsqueeze(1),feed_dict.mask[:,0].unsqueeze(-1))
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
 
